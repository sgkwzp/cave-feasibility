"""Frozen-model transfer, matched fixed ROI, and unchanged normal-only scoring."""
import json,time
import numpy as np
import torch
from common import ROOT,read,dump,sha
from observer import Observer,pool_vector
from evaluate import density,finite

EXT=ROOT/'remaining-tasks'

def main():
    torch.set_num_threads(4);start=time.monotonic();cfg=read(EXT/'config.json')
    import detect
    detect.ROOT=EXT
    detect.main()
    model=Observer().cuda().eval();ckpt=torch.load(ROOT/cfg['observer_checkpoint'],map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=True);model.requires_grad_(False)
    z=np.load(EXT/'text-embeddings.npz');text=dict(zip(z['strings'].tolist(),z['vectors']))
    dest=EXT/'embeddings';dest.mkdir(exist_ok=True);videos=read(EXT/'feature-index.json');coverage=[];checks=[]
    for iv,v in enumerate(videos):
        d=read(EXT/'roi'/(v['video']+'.json'));masks={(x['uid'],tuple(x['names'])):np.array(x['masks'],np.float32) for x in d}
        vectors={k:[] for k in ['Global','ROI','Learned']}
        for i,r in enumerate(v['rows']):
            arrays=[np.load(EXT/'features'/(uid+'.npz')) for uid in r['frame_uids']]
            patch=torch.from_numpy(np.stack([a['patch'] for a in arrays])).cuda().float()
            valid=torch.from_numpy(np.stack([a['valid'] for a in arrays])).cuda().float()
            coords=torch.from_numpy(np.stack([a['coords'] for a in arrays])).cuda().float()
            query=torch.from_numpy(np.stack([text['a photo of '+n.lower()] for n in r['objects']]))[None].expand(3,-1,-1).cuda()
            rr=np.stack([masks[(uid,tuple(r['objects']))] for uid in r['frame_uids']]);available=rr.sum(-1)>0
            for f,s in np.argwhere(~available):rr[f,s]=arrays[f]['valid']
            rr=rr/rr.sum(-1,keepdims=True);roi=torch.from_numpy(rr).cuda();uniform=valid/valid.sum(-1,keepdim=True)
            assert np.isfinite(rr).all() and np.allclose(rr.sum(-1),1,atol=1e-6)
            with torch.no_grad():
                g=pool_vector(patch,uniform[:,None,:].expand(-1,2,-1),coords).mean(0).cpu().numpy()
                o=pool_vector(patch,roi,coords).mean(0).cpu().numpy()
                l=model(patch,query,valid,coords)[0].mean(0).cpu().numpy()
            for k,x in [('Global',g),('ROI',o),('Learned',l)]:vectors[k].append(x)
            if i==0:
                pp=patch.cpu().numpy();cc=coords.cpu().numpy();slot=np.einsum('bsn,bnd->bsd',rr,pp)
                slot/=np.maximum(np.linalg.norm(slot,axis=-1,keepdims=True),1e-12)
                centers=np.einsum('bsn,bnd->bsd',rr,cc)*.25
                expected=np.concatenate([slot.reshape(3,-1),centers.reshape(3,-1)],axis=-1).mean(0)
                err=float(np.max(np.abs(expected-o)));assert err<3e-6,err;checks.append({'video':v['video'],'max_error':err})
            coverage.append({k:r[k] for k in ['id','task','video','step','split','kind']}|{'available_slots':int(available.sum()),'total_slots':6})
            for a in arrays:a.close()
        np.savez_compressed(dest/(v['video']+'.npz'),ids=[r['id'] for r in v['rows']],**{k:np.stack(x) for k,x in vectors.items()})
        if (iv+1)%20==0:print('EXTENSION_POOLED',iv+1,len(videos),flush=True)
    dump(EXT/'coverage.json',coverage)
    for task in cfg['tasks']:
        rows=[];vectors={k:[] for k in ['Global','ROI','Learned']}
        for v in videos:
            if v['task']!=task:continue
            z=np.load(dest/(v['video']+'.npz'));assert z['ids'].tolist()==[r['id'] for r in v['rows']]
            rows.extend(v['rows'])
            for k in vectors:vectors[k].append(z[k])
        baseline={r['id']:r for r in read(EXT/'baseline'/task/'scores.json')}
        scores={k:np.array([baseline[r['id']]['scores'][k] for r in rows]) for k in ['AEM','Duration']}
        raw={k:np.array([baseline[r['id']]['raw'][k] for r in rows]) for k in scores};fits={}
        for k,x in vectors.items():raw[k],scores[k],fits[k]=density(np.concatenate(x),rows)
        scores['AEM+Duration']=(scores['AEM']+scores['Duration'])/2
        for k in vectors:
            scores['AEM+'+k]=(scores['AEM']+scores[k])/2
            scores['AEM+Duration+'+k]=(scores['AEM']+scores['Duration']+scores[k])/3
        result=[dict(r,raw={k:float(x[i]) for k,x in raw.items()},scores={k:float(x[i]) for k,x in scores.items()}) for i,r in enumerate(rows)]
        dump(EXT/'results'/task/'scores.json',result);dump(EXT/'results'/task/'fits.json',finite(fits))
        print('EXTENSION_SCORED',task,len(rows),flush=True)
    dump(EXT/'score-verification.json',{'videos':len(videos),'segments':len(coverage),'pooling_checks':checks,
        'checkpoint_sha256':sha(ROOT/cfg['observer_checkpoint']),'strict_loading':True,'new_training_runs':0,
        'source_sha256':sha(__file__),'seconds':time.monotonic()-start})
    dump(EXT/'status.json',{'phase':'remaining_tasks_scored_pending_analysis','new_training_runs':0})

if __name__=='__main__':main()
