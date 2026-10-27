"""Same frozen EVA02 observation frames for three additional tasks."""
import hashlib,json,shutil,sys,time,zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from common import ROOT,PROJECT,OEV,read,dump,sha
sys.path.insert(0,str(OEV/'cache/deps'))
import open_clip
from extract import letterbox,geometry

EXT=ROOT/'remaining-tasks'

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);start=time.monotonic()
    videos=read(EXT/'manifest.json')
    weights=next((Path.home()/'.cache/huggingface/hub/models--timm--eva02_large_patch14_clip_336.merged2b_s6b_b61k/snapshots').glob('*/open_clip_model.safetensors'))
    old=read(ROOT/'feature-signature.json');assert sha(weights)==old['weights_sha256']
    model,_,_=open_clip.create_model_and_transforms('EVA02-L-14-336',pretrained=str(weights),precision='fp16',device='cuda')
    model.eval().requires_grad_(False)
    strings=sorted(set('a photo of '+n.lower() for pair in read(EXT/'queries.json').values() for n in pair))
    tokenizer=open_clip.get_tokenizer('EVA02-L-14-336');text=[]
    with torch.no_grad():
        for i in range(0,len(strings),32):
            t=model.encode_text(tokenizer(strings[i:i+32]).cuda()).float();text.append(torch.nn.functional.normalize(t,dim=-1).cpu().numpy())
    np.savez(EXT/'text-embeddings.npz',strings=strings,vectors=np.concatenate(text))
    mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
    feat=EXT/'features';images_dir=EXT/'cache/tail-images';video_dir=EXT/'cache/videos'
    for p in [feat,images_dir,video_dir]:p.mkdir(parents=True,exist_ok=True)
    index=[];framecount=0;encode_seconds=0
    for iv,v in enumerate(videos):
        meta_path=feat/(v['video']+'.json')
        if meta_path.exists():index.append(read(meta_path));continue
        begin=time.monotonic();mp4=video_dir/(v['video']+'.mp4');zp=PROJECT/v['zip']
        with zipfile.ZipFile(zp) as z,z.open(v['member']) as src,mp4.open('wb') as out:shutil.copyfileobj(src,out)
        cap=cv2.VideoCapture(str(mp4));assert cap.isOpened(),mp4
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert fps>0 and count>0
        frameids=[]
        for r in v['rows']:
            assert r['end']<=count/fps+.6,(r['id'],r['end'],count/fps)
            upper=count-1
            if r['next_action_start'] is not None:upper=min(upper,int(np.floor((r['next_action_start']-1e-6)*fps)))
            frameids.append([min(upper,max(0,int(round(t*fps)))) for t in r['old_times']])
        entries=[];ims=[];lookup={}
        for ix in sorted(set(np.array(frameids).ravel().tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,im=cap.read();assert ok,(v['video'],ix)
            h,w=im.shape[:2];sc=min(1.,854/w,480/h);im=cv2.resize(im,(round(w*sc),round(h*sc)),interpolation=cv2.INTER_AREA)
            uid=hashlib.sha256(f"{v['video']}:{ix}".encode()).hexdigest()[:20];lookup[ix]=uid
            cv2.imwrite(str(images_dir/(uid+'.jpg')),im,[cv2.IMWRITE_JPEG_QUALITY,95])
            entries.append({'uid':uid,'type':'tail','frame_index':ix});ims.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        cap.release();assert mp4.resolve().parent==video_dir.resolve();mp4.unlink()
        ts=time.monotonic()
        for b in range(0,len(ims),8):
            boxed=[letterbox(im) for im in ims[b:b+8]]
            pixels=torch.from_numpy(np.stack([x[0] for x in boxed])).permute(0,3,1,2).cuda().float()/255
            with torch.no_grad():z=model.visual.trunk.forward_features(((pixels-mean)/std).half()).float().cpu().numpy()
            assert z.shape[1:]==(577,1024) and np.isfinite(z).all()
            for j,(_,g) in enumerate(boxed):
                e=entries[b+j];valid,coords=geometry(g);e['geometry']=g
                np.savez(feat/(e['uid']+'.npz'),patch=z[j,1:].astype('float16'),valid=valid,coords=coords)
        enc=time.monotonic()-ts;encode_seconds+=enc;framecount+=len(ims)
        meta={**v,'rows':[dict(r,frame_uids=[lookup[ix] for ix in frameids[i]]) for i,r in enumerate(v['rows'])],
            'entries':entries,'fps':fps,'count':count,'seconds':time.monotonic()-begin,'encode_seconds':enc}
        dump(meta_path,meta);index.append(meta)
        status={'phase':'extracting_remaining','completed_videos':iv+1,'total_videos':len(videos),'frames_this_run':framecount,
                'seconds':time.monotonic()-start,'last_video':v['video']}
        dump(EXT/'status.json',status);print(json.dumps(status),flush=True)
    dump(EXT/'feature-index.json',index)
    dump(EXT/'feature-signature.json',{'weights_sha256':sha(weights),'parent_feature_signature_sha256':sha(ROOT/'feature-signature.json'),
         'manifest_sha256':sha(EXT/'manifest.json'),'source_sha256':sha(__file__),'seconds':time.monotonic()-start,
         'encoding_seconds':encode_seconds,'frames':sum(len(v['entries']) for v in index)})
    print('REMAINING_EXTRACTION_COMPLETE',flush=True)

if __name__=='__main__':main()
