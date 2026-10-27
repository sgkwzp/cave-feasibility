"""Frozen EVA02 patches from exact teacher images and locked OEV tail frames."""
import argparse, collections, hashlib, json, os, shutil, time, zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
import open_clip
from common import ROOT,PROJECT,OEV,read,dump,sha

def letterbox(im, size=336):
    h,w=im.shape[:2];scale=min(size/w,size/h);ww,hh=round(w*scale),round(h*scale)
    dx,dy=(size-ww)//2,(size-hh)//2
    out=np.zeros((size,size,3),np.uint8)
    out[dy:dy+hh,dx:dx+ww]=cv2.resize(im,(ww,hh),interpolation=cv2.INTER_AREA)
    return out,dict(width=w,height=h,ww=ww,hh=hh,dx=dx,dy=dy)

def mask_box(box,geom):
    x0,y0,x1,y1=np.asarray(box,float)
    x0,x1=(np.array([x0,x1])*geom['ww']+geom['dx'])/14
    y0,y1=(np.array([y0,y1])*geom['hh']+geom['dy'])/14
    wx=np.maximum(0,np.minimum(np.arange(24)+1,x1)-np.maximum(np.arange(24),x0))
    wy=np.maximum(0,np.minimum(np.arange(24)+1,y1)-np.maximum(np.arange(24),y0))
    return (wy[:,None]*wx[None,:]).reshape(-1).astype('float32')

def geometry(g):
    valid=mask_box([0,0,1,1],g)
    yy,xx=np.meshgrid((np.arange(24)+.5)*14,(np.arange(24)+.5)*14,indexing='ij')
    coords=np.stack([(xx-g['dx'])/g['ww'],(yy-g['dy'])/g['hh']],-1).reshape(-1,2)
    return valid,np.clip(coords,0,1).astype('float32')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    cfg=read(ROOT/'execution.json');videos=read(ROOT/'manifest.json');teachers=read(ROOT/'supervision.json')
    weights=next((Path.home()/'.cache/huggingface/hub/models--timm--eva02_large_patch14_clip_336.merged2b_s6b_b61k/snapshots').glob('*/open_clip_model.safetensors'))
    started=time.monotonic()
    model,_,_=open_clip.create_model_and_transforms('EVA02-L-14-336',pretrained=str(weights),precision='fp16',device='cuda')
    model.eval().requires_grad_(False);tokenizer=open_clip.get_tokenizer('EVA02-L-14-336')
    mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
    feature_dir=ROOT/'features';feature_dir.mkdir(exist_ok=True)
    image_dir=ROOT/'cache/tail-images';image_dir.mkdir(parents=True,exist_ok=True)
    signature=dict(encoder='EVA02-L-14-336',weights_sha256=sha(weights),
        manifest_sha256=sha(ROOT/'manifest.json'),supervision_sha256=sha(ROOT/'supervision.json'),
        source_sha256=sha(__file__),torch=torch.__version__,open_clip=open_clip.__version__,device=torch.cuda.get_device_name(),
        geometry='336 square letterbox; original-relative patch coordinates',precision='fp16 stored patches; float32 downstream',
        frames='exact OEV old tail indices; exact published teacher RGB archive keys')
    if (ROOT/'feature-signature.json').exists():assert read(ROOT/'feature-signature.json')==signature,'cache signature changed'
    dump(ROOT/'feature-signature.json',signature)
    strings=set('a photo of '+name.lower() for q in read(ROOT/'queries.json').values() for name in q)
    for t in teachers:
        strings.update('a photo of '+name.lower() for name in t['names'])
        strings.update(t['attribute_sentence']);strings.update(t['relation_sentence'])
    strings=sorted(strings);vectors=[]
    with torch.no_grad():
        for i in range(0,len(strings),32):
            z=model.encode_text(tokenizer(strings[i:i+32]).cuda()).float()
            vectors.append(torch.nn.functional.normalize(z,dim=-1).cpu().numpy())
    np.savez(ROOT/'text-embeddings.npz',strings=np.array(strings),vectors=np.concatenate(vectors))
    archives={t:np.load(PROJECT/f'code/AEM/data/egoper/effect_frames/effect_frames_{t}.npz') for t in ['quesadilla','pinwheels']}
    teacher_groups=collections.defaultdict(list)
    for t in teachers:teacher_groups[t['video']].append(t)
    if args.smoke:videos=[next(v for v in videos if v['task']==t and v['split']=='train') for t in ['quesadilla','pinwheels']]
    allmeta=[];encode_seconds=0.;frames=0
    for iv,v in enumerate(videos):
        meta_path=feature_dir/(v['video']+'.json')
        if meta_path.exists():allmeta.append(read(meta_path));continue
        vstart=time.monotonic();entries=[];images=[]
        for t in teacher_groups[v['video']]:
            key=t['teacher_image_path'];assert key in archives[v['task']],key
            im=archives[v['task']][key];assert im.dtype==np.uint8 and im.ndim==3
            entries.append(dict(uid=hashlib.sha256(t['id'].encode()).hexdigest()[:20],type='teacher',teacher=t))
            images.append(im)
        old=read(OEV/'features'/(v['video']+'.json'))
        orig=np.load(OEV/'features'/(v['video']+'.npz'))
        assert orig['ids'].tolist()==[r['id'] for r in v['rows']]
        indices=orig['frame_indices'][:,:3];unique=sorted(set(indices.reshape(-1).tolist()))
        folder=ROOT/'cache/videos';folder.mkdir(exist_ok=True)
        path=folder/(v['video']+'.mp4')
        zip_path=Path(v['zip'])
        candidates=[PROJECT/zip_path,PROJECT/'data'/zip_path,PROJECT/'code/EgoPER'/zip_path]
        zp=next((p for p in candidates if p.exists()),None);assert zp is not None,candidates
        with zipfile.ZipFile(zp) as archive,archive.open(v['member']) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
        cap=cv2.VideoCapture(str(path));assert cap.isOpened()
        assert abs(cap.get(cv2.CAP_PROP_FPS)-old['fps'])<1e-6
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==old['count']
        lookup={}
        for ix in unique:
            cap.set(cv2.CAP_PROP_POS_FRAMES,ix);ok,im=cap.read();assert ok,(v['video'],ix)
            h,w=im.shape[:2];sc=min(1.,854/w,480/h)
            im=cv2.resize(im,(round(w*sc),round(h*sc)),interpolation=cv2.INTER_AREA)
            uid=hashlib.sha256(f"{v['video']}:{ix}".encode()).hexdigest()[:20]
            lookup[ix]=uid
            cv2.imwrite(str(image_dir/(uid+'.jpg')),im,[cv2.IMWRITE_JPEG_QUALITY,95])
            entries.append(dict(uid=uid,type='tail',frame_index=ix));images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        cap.release();assert path.resolve().parent==folder.resolve();path.unlink()
        ti=time.monotonic()
        for start in range(0,len(images),8):
            boxed=[letterbox(im) for im in images[start:start+8]]
            pixels=torch.from_numpy(np.stack([b[0] for b in boxed])).permute(0,3,1,2).cuda().float()/255
            with torch.no_grad():z=model.visual.trunk.forward_features(((pixels-mean)/std).half()).float().cpu().numpy()
            assert z.shape[1:]==(577,1024),z.shape
            assert np.isfinite(z).all()
            for j,(_,geom) in enumerate(boxed):
                e=entries[start+j];valid,coords=geometry(geom);masks=np.zeros((2,576),np.float32)
                if e['type']=='teacher':
                    for k,b in enumerate(e['teacher']['boxes_cxcywh_normalized']):
                        if b is not None:
                            cx,cy,w,h=b;masks[k]=mask_box(np.clip([cx-w/2,cy-h/2,cx+w/2,cy+h/2],0,1),geom)
                np.savez(feature_dir/(e['uid']+'.npz'),patch=z[j,1:].astype('float16'),cls=z[j,0].astype('float16'),
                    valid=valid,coords=coords,teacher_masks=masks)
                e['geometry']=geom
        elapsed_encode=time.monotonic()-ti;encode_seconds+=elapsed_encode;frames+=len(images)
        meta=dict(video=v['video'],task=v['task'],split=v['split'],entries=entries,
            rows=[dict(r,frame_uids=[lookup[int(ix)] for ix in indices[i]]) for i,r in enumerate(v['rows'])],
            seconds=time.monotonic()-vstart,encode_seconds=elapsed_encode)
        dump(meta_path,meta);allmeta.append(meta)
        status=dict(phase='extracting_smoke' if args.smoke else 'extracting',completed_videos=iv+1,total_videos=len(videos),
            last_video=v['video'],frames_this_run=frames,seconds=time.monotonic()-started,
            encoding_seconds=encode_seconds,peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20)
        dump(ROOT/'status.json',status);print(json.dumps(status),flush=True)
    dump(ROOT/('build/smoke-extraction.json' if args.smoke else 'feature-index.json'),allmeta)
    print('SMOKE_COMPLETE' if args.smoke else 'EXTRACTION_COMPLETE',flush=True)

if __name__=='__main__':main()
