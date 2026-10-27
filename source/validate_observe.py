"""Stream both observation arms; reuse identical old frames/detections, save vectors."""
import argparse,hashlib,json,shutil,sys,time,zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from common import ROOT,PROJECT,OEV,read,dump,sha
sys.path.insert(0,str(OEV/'cache/deps'))
import open_clip
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from extract import letterbox,geometry,mask_box
from detect import words
from observer import pool_vector

OUT=ROOT/'deployment-mechanism'

def main(limit=None,worker=0,workers=1):
    assert 0<=worker<workers
    torch.set_num_threads(4);cv2.setNumThreads(1);start=time.monotonic()
    frozen=read(OUT/'frozen-inputs.json')
    assert sha(OUT/'manifest.json')==frozen['manifest_sha256']
    assert sha(OUT/'protocol.md')==frozen['protocol_sha256']
    for p,h in frozen['files'].items():assert sha(PROJECT/p)==h,p
    weights=next((Path.home()/'.cache/huggingface/hub/models--timm--eva02_large_patch14_clip_336.merged2b_s6b_b61k/snapshots').glob('*/open_clip_model.safetensors'))
    assert sha(weights)==read(ROOT/'feature-signature.json')['weights_sha256']
    model,_,_=open_clip.create_model_and_transforms('EVA02-L-14-336',pretrained=str(weights),precision='fp16',device='cuda')
    model.eval().requires_grad_(False)
    md=OEV/'cache/grounding-dino-tiny'
    processor=AutoProcessor.from_pretrained(str(md),local_files_only=True)
    processor.image_processor.size={'shortest_edge':480,'longest_edge':854}
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(str(md),local_files_only=True,disable_custom_kernels=True).cuda().eval().requires_grad_(False)
    mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
    folders={k:OUT/k for k in ['embeddings','observations','cache/videos','cache/images']}
    for p in folders.values():p.mkdir(parents=True,exist_ok=True)
    oldmeta={v['video']:(base,v) for base in [ROOT,ROOT/'remaining-tasks'] for v in read(base/'feature-index.json')}
    videos=read(OUT/'manifest.json');totals={'new_frames':0,'reused_frames':0,'new_detections':0,'reused_detections':0}
    sig={'manifest_sha256':sha(OUT/'manifest.json'),'source_sha256':sha(__file__),'eva_sha256':sha(weights),
         'detector_sha256':sha(md/'model.safetensors'),'training_runs':0}
    if (OUT/'observation-signature.json').exists():
        prior=read(OUT/'observation-signature.json')
        assert all(prior[k]==v for k,v in sig.items() if k!='source_sha256')
    else:dump(OUT/'observation-signature.json',sig)
    dump(OUT/f'build/worker-{worker}-signature.json',dict(sig,worker=worker,workers=workers,
         revision='Only disjoint video sharding and per-worker progress; image/model/scoring operations unchanged'))
    for iv,v in enumerate(videos):
        if iv%workers!=worker:continue
        dest=folders['embeddings']/(v['video']+'.npz');meta_path=folders['observations']/(v['video']+'.json')
        if dest.exists() and meta_path.exists():continue
        begin=time.monotonic();base,old=oldmeta[v['video']]
        geoms={e['uid']:e['geometry'] for e in old['entries'] if e['type']=='tail'}
        olddet={(r['uid'],tuple(r['names'])):r for r in read(base/'roi'/(v['video']+'.json'))}
        mp4=folders['cache/videos']/(v['video']+'.mp4')
        with zipfile.ZipFile(PROJECT/v['zip']) as z,z.open(v['member']) as src,mp4.open('wb') as dst:shutil.copyfileobj(src,dst)
        cap=cv2.VideoCapture(str(mp4));assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert fps>0 and count>0
        rows=[];requests={};frames={};newimages=[];newuids=[];entries={}
        for r in v['rows']:
            upper=count-1 if r['upper'] is None else min(count-1,int(np.floor(r['upper']*fps+1e-6)))
            ix=[min(upper,max(0,int(round(t*fps)))) for t in r['times']]
            uids=[hashlib.sha256(f"{v['video']}:{j}".encode()).hexdigest()[:20] for j in ix]
            rr=dict(r,frame_indices=ix,frame_uids=uids);rows.append(rr)
            for uid,j in zip(uids,ix):
                requests[(uid,tuple(r['objects']))]=None
                entries[uid]={'uid':uid,'frame_index':j}
        # One real old observation per recording verifies the identical pooling path.
        anchor=old['rows'][0]
        for uid in anchor['frame_uids']:
            requests[(uid,tuple(anchor['objects']))]=None
            if uid not in entries:entries[uid]={'uid':uid,'frame_index':next(e['frame_index'] for e in old['entries'] if e['uid']==uid)}
        for uid,e in sorted(entries.items(),key=lambda p:p[1]['frame_index']):
            cached=base/'features'/(uid+'.npz')
            if uid in geoms and cached.exists():
                with np.load(cached) as z:frames[uid]={k:z[k].copy() for k in ['patch','valid','coords']}
                e.update(geometry=geoms[uid],image=str(base/'cache/tail-images'/(uid+'.jpg')),reused=True)
                totals['reused_frames']+=1;continue
            cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_index']);ok,im=cap.read();assert ok,(v['video'],e)
            h,w=im.shape[:2];sc=min(1.,854/w,480/h);im=cv2.resize(im,(round(w*sc),round(h*sc)),interpolation=cv2.INTER_AREA)
            path=folders['cache/images']/(uid+'.jpg');cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,95])
            boxed,g=letterbox(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));e.update(geometry=g,image=str(path),reused=False)
            newimages.append(boxed);newuids.append(uid);totals['new_frames']+=1
        cap.release();assert mp4.resolve().parent==folders['cache/videos'].resolve();mp4.unlink()
        for i in range(0,len(newuids),8):
            pixel=torch.from_numpy(np.stack(newimages[i:i+8])).permute(0,3,1,2).cuda().float()/255
            with torch.no_grad():z=model.visual.trunk.forward_features(((pixel-mean)/std).half()).float().cpu().numpy()
            assert z.shape[1:]==(577,1024) and np.isfinite(z).all()
            for j,uid in enumerate(newuids[i:i+8]):
                valid,coords=geometry(entries[uid]['geometry'])
                frames[uid]={'patch':z[j,1:].astype('float16'),'valid':valid,'coords':coords}
        del newimages
        masks={};detections=[];pending=[]
        for key in requests:
            if key in olddet:
                d=olddet[key];masks[key]=np.array(d['masks'],np.float32)
                detections.append({k:d[k] for k in ['uid','names','selected','detections']}|{'reused':True})
                totals['reused_detections']+=1
            else:pending.append(key)
        for i in range(0,len(pending),4):
            batch=pending[i:i+4];ims=[Image.open(entries[uid]['image']).convert('RGB') for uid,_ in batch]
            prompts=[' . '.join(n.lower() for n in names)+' .' for _,names in batch]
            inp=processor(images=ims,text=prompts,padding=True,return_tensors='pt').to('cuda')
            with torch.no_grad():pred=detector(**inp)
            dets=processor.post_process_grounded_object_detection(pred,inp.input_ids,box_threshold=.25,text_threshold=.2,
                 target_sizes=[(im.height,im.width) for im in ims])
            for key,im,d in zip(batch,ims,dets):
                uid,names=key;boxes=[{'label':l,'score':float(s),'box':b.cpu().tolist()} for l,s,b in zip(d['labels'],d['scores'],d['boxes'])]
                mm=[];chosen=[]
                for name in names:
                    nw=words(name);candidates=[]
                    for b in boxes:
                        bw=words(b['label']);overlap=len(nw&bw)/max(1,len(nw|bw))
                        if overlap>=.5:candidates.append((overlap,b['score'],b))
                    if candidates:
                        b=max(candidates,key=lambda x:(x[0],x[1]))[2]
                        xy=np.clip(np.array(b['box'])/np.array([im.width,im.height,im.width,im.height]),0,1)
                        mm.append(mask_box(xy,entries[uid]['geometry']));chosen.append(b)
                    else:mm.append(np.zeros(576,np.float32));chosen.append(None)
                masks[key]=np.stack(mm);detections.append({'uid':uid,'names':names,'selected':chosen,'detections':boxes,'reused':False})
                totals['new_detections']+=1
            del inp,pred
        def pool(r):
            pp=np.stack([frames[u]['patch'] for u in r['frame_uids']]).astype('float32')
            cc=np.stack([frames[u]['coords'] for u in r['frame_uids']])
            mm=np.stack([masks[(u,tuple(r['objects']))] for u in r['frame_uids']]);available=mm.sum(-1)>0
            for f,s in np.argwhere(~available):mm[f,s]=frames[r['frame_uids'][f]]['valid']
            mm/=mm.sum(-1,keepdims=True)
            with torch.no_grad():x=pool_vector(torch.from_numpy(pp).cuda(),torch.from_numpy(mm).cuda(),torch.from_numpy(cc).cuda()).mean(0).cpu().numpy()
            assert np.isfinite(x).all();return x,int(available.sum())
        av,_=pool(anchor)
        oldpath=(ROOT/'follow-up/embeddings' if base==ROOT else base/'embeddings')/(v['video']+'.npz')
        with np.load(oldpath) as z:expected=z['ROI'][z['ids'].tolist().index(anchor['id'])]
        err=float(np.max(abs(av-expected)));assert err<3e-6,(v['video'],err)
        vectors=[]
        for r in rows:
            x,avail=pool(r);vectors.append(x);r['available_slots']=avail
        np.savez_compressed(dest,ids=[r['id'] for r in rows],ROI=np.stack(vectors))
        dump(meta_path,{'video':v['video'],'task':v['task'],'split':v['split'],'rows':rows,'entries':list(entries.values()),
             'detections':detections,'fps':fps,'frame_count':count,'anchor_max_error':err,'seconds':time.monotonic()-begin})
        status={'phase':'observing','worker':worker,'completed_videos':len(list(folders['observations'].glob('*.json'))),'total_videos':len(videos),'last_video':v['video'],
                'seconds':time.monotonic()-start,'totals_this_run':totals,'last_video_seconds':time.monotonic()-begin}
        dump(OUT/f'build/worker-{worker}-status.json',status);print(json.dumps(status),flush=True)
        if limit is not None and iv+1>=limit:return
    dump(OUT/f'build/worker-{worker}-status.json',{'phase':'worker_complete','worker':worker,'seconds':time.monotonic()-start,'totals_this_run':totals})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();main(a.limit,a.worker,a.workers)
