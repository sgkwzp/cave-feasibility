"""Additional matched ROI comparison, preserving every original G1 output."""
import collections, json, time
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from common import ROOT, TASKS, read, dump, sha
from evaluate import density, smooth_score, finite
from observer import pool_vector

OUT=ROOT/'follow-up'

def centroid_density(x,rows):
    tr=np.array([r['split']=='train' for r in rows]); steps=np.array([r['step'] for r in rows])
    vid=np.array([r['video'] for r in rows]); scaler=StandardScaler().fit(x[tr]); z=scaler.transform(x)
    raw=np.zeros(len(rows)); score=raw.copy(); fits={}
    for a in sorted(set(steps)):
        m=steps==a; ref=np.flatnonzero(m&tr)
        model=LedoitWolf().fit(z[ref]); raw[m]=model.mahalanobis(z[m])
        for v in np.unique(vid[ref]):
            held=ref[vid[ref]==v]; fit=ref[vid[ref]!=v]
            raw[held]=LedoitWolf().fit(z[fit]).mahalanobis(z[held])
        score[m]=smooth_score(raw[m],raw[ref])
        fits[str(a)]={'mean':model.location_.tolist(),'precision':model.precision_.tolist(),'n':len(ref)}
    return raw,score,{'steps':fits,'scale':scaler.scale_.tolist(),'mean':scaler.mean_.tolist(),
                     'reference':'train-only standardization; LOO-video covariance; empirical calibration'}

def main():
    torch.set_num_threads(4); start=time.monotonic()
    frozen=read(OUT/'frozen-inputs.json')
    for p,h in frozen['original'].items(): assert sha(ROOT/p)==h,p
    import detect
    original_dump=detect.dump
    def routed_dump(path,obj):
        original_dump(OUT/'status.json' if path==ROOT/'status.json' else path,obj)
    detect.dump=routed_dump
    detect.main()
    videos=read(ROOT/'feature-index.json'); coverage=[]; checks=[]
    dest=OUT/'embeddings';dest.mkdir(exist_ok=True)
    for iv,v in enumerate(videos):
        detections=read(ROOT/'roi'/(v['video']+'.json'))
        masks={(d['uid'],tuple(d['names'])):np.array(d['masks'],np.float32) for d in detections}
        vectors=[]
        for r in v['rows']:
            arrays=[np.load(ROOT/'features'/(uid+'.npz')) for uid in r['frame_uids']]
            patch=np.stack([a['patch'] for a in arrays]).astype(np.float32)
            valid=np.stack([a['valid'] for a in arrays]).astype(np.float32)
            coords=np.stack([a['coords'] for a in arrays]).astype(np.float32)
            rr=np.stack([masks[(uid,tuple(r['objects']))] for uid in r['frame_uids']])
            available=rr.sum(-1)>0
            for f,s in np.argwhere(~available): rr[f,s]=valid[f]
            rr=rr/rr.sum(-1,keepdims=True)
            assert np.isfinite(rr).all() and np.allclose(rr.sum(-1),1,atol=1e-6)
            assert np.max(rr*(valid[:,None,:]==0))==0
            with torch.no_grad():
                vec=pool_vector(torch.from_numpy(patch),torch.from_numpy(rr),torch.from_numpy(coords)).mean(0).numpy()
            vectors.append(vec)
            # Independent NumPy reconstruction on one predetermined segment per video.
            if len(vectors)==1:
                slot=np.einsum('bsn,bnd->bsd',rr,patch)
                slot/=np.maximum(np.linalg.norm(slot,axis=-1,keepdims=True),1e-12)
                center=np.einsum('bsn,bnd->bsd',rr,coords)*.25
                check=np.concatenate([slot.reshape(3,-1),center.reshape(3,-1)],axis=1).mean(0)
                err=float(np.max(np.abs(vec-check)));assert err<3e-6,err
                checks.append({'video':v['video'],'max_abs_error':err})
            coverage.append({k:r[k] for k in ['id','video','task','split','kind','step']}|
                {'available_slots':int(available.sum()),'total_slots':6,'objects':r['objects']})
            for a in arrays:a.close()
        np.savez_compressed(dest/(v['video']+'.npz'),ids=[r['id'] for r in v['rows']],ROI=np.stack(vectors))
        if (iv+1)%20==0:print('ROI_POOLED',iv+1,len(videos),flush=True)
    dump(OUT/'coverage.json',coverage)
    for task in TASKS:
        rows=[]; roi=[]; learned=[]
        for v in videos:
            if v['task']!=task:continue
            old=np.load(ROOT/'embeddings'/(v['video']+'.npz'));new=np.load(dest/(v['video']+'.npz'))
            assert old['ids'].tolist()==new['ids'].tolist()==[r['id'] for r in v['rows']]
            rows.extend(v['rows']);roi.append(new['ROI']);learned.append(old['Learned_20261027'])
        roi=np.concatenate(roi);learned=np.concatenate(learned); fits={};raw={};scores={}
        raw['ROI'],scores['ROI'],fits['ROI']=density(roi,rows)
        for name,x in [('ROI-centroid',roi),('Learned-centroid',learned)]:
            raw[name],scores[name],fits[name]=centroid_density(x[:,-4:]/.25,rows)
        records=[{'id':r['id'],'scores':{k:float(s[i]) for k,s in scores.items()},
                  'raw':{k:float(s[i]) for k,s in raw.items()}} for i,r in enumerate(rows)]
        dump(OUT/'roi-results'/task/'scores.json',records);dump(OUT/'roi-results'/task/'fits.json',finite(fits))
        print('ROI_SCORED',task,len(rows),flush=True)
    for p,h in frozen['original'].items():assert sha(ROOT/p)==h,p
    dump(OUT/'roi-verification.json',{'original_inputs_unchanged':True,'pooling_checks':checks,
        'segments':len(coverage),'videos':len(videos),'seconds':time.monotonic()-start,
        'source_sha256':sha(__file__),'config_sha256':sha(OUT/'config.json')})
    dump(OUT/'status.json',{'phase':'ROI_completed_pending_analysis','seconds':time.monotonic()-start})
    print('ROI_COMPLETE',flush=True)

if __name__=='__main__':main()
