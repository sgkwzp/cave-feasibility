"""Matched observation scores, normal-only density fitting, paired video bootstrap."""
import argparse,collections,json,time
import numpy as np
import torch
from scipy.special import log_ndtr,logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score,average_precision_score
from common import ROOT,OEV,TASKS,read,dump,sha
from observer import Observer,pool_vector

def finite(x):
    if isinstance(x,dict):return {str(k):finite(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [finite(v) for v in x]
    if isinstance(x,np.ndarray):return finite(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x

def smooth_score(x,ref):
    ref=np.asarray(ref);sd=float(np.std(ref,ddof=1)) if len(ref)>1 else 0
    iq=float(np.subtract(*np.percentile(ref,[75,25]))/1.349)
    scale=min(sd,iq) if min(sd,iq)>1e-10 else max(sd,iq,1e-6)
    h=max(.9*scale*len(ref)**(-.2),1e-6)
    return -(logsumexp(log_ndtr((ref[None,:]-np.asarray(x)[:,None])/h),axis=1)-np.log(len(ref)))

def auc(y,s,w):
    pos=(w*y).sum();neg=(w*(~y)).sum()
    if pos<=0 or neg<=0:return np.nan
    order=np.argsort(s,kind='stable');y=y[order];s=s[order];w=w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
    pp=np.add.reduceat(w*y,starts);nn=np.add.reduceat(w*(~y),starts)
    return float((pp*(np.cumsum(nn)-nn+.5*nn)).sum()/(pos*neg))

def density(x,rows):
    x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    tr=np.array([r['split']=='train' for r in rows]);steps=np.array([r['step'] for r in rows]);vid=np.array([r['video'] for r in rows])
    pca=PCA(n_components=32,svd_solver='full').fit(x[tr]);z=pca.transform(x)
    raw=np.zeros(len(rows));fits={}
    for a in sorted(set(steps)):
        ix=np.flatnonzero(steps==a);reference=np.flatnonzero((steps==a)&tr)
        assert len(np.unique(vid[reference]))>=5,(a,len(reference))
        model=LedoitWolf().fit(z[reference]);raw[ix]=model.mahalanobis(z[ix])
        for video in np.unique(vid[reference]):
            held=reference[vid[reference]==video];fit=reference[vid[reference]!=video]
            loo=LedoitWolf().fit(z[fit]);raw[held]=loo.mahalanobis(z[held])
        fits[str(a)]={'n':len(reference),'videos':len(np.unique(vid[reference])),'shrinkage':float(model.shrinkage_),
            'mean':model.location_.tolist(),'precision':model.precision_.tolist()}
    score=np.zeros(len(rows))
    for a in sorted(set(steps)):
        m=steps==a;score[m]=smooth_score(raw[m],raw[m&tr])
    return raw,score,{'steps':fits,'explained_variance':pca.explained_variance_ratio_.tolist(),
        'reference':'leave-one-training-video Gaussian distances; PCA fit once on normal training; empirical calibration'}

def embeddings(probe=False):
    cfg=read(ROOT/'execution.json');videos=read(ROOT/'feature-index.json')
    z=np.load(ROOT/'text-embeddings.npz');text=dict(zip(z['strings'].tolist(),z['vectors']))
    models=[]
    for seed in cfg['seeds']:
        m=Observer().cuda().eval();checkpoint=torch.load(ROOT/'models'/f'{seed}.pt',map_location='cpu')
        m.load_state_dict(checkpoint['state_dict'],strict=True);m.requires_grad_(False);models.append(m)
    dest=ROOT/'embeddings';dest.mkdir(exist_ok=True);coverage=[]
    for iv,v in enumerate(videos):
        path=dest/(v['video']+'.npz')
        if path.exists() and (probe or 'ROI' in np.load(path).files):continue
        detections=[] if probe else read(ROOT/'roi'/(v['video']+'.json'))
        masks={(d['uid'],tuple(d['names'])):np.array(d['masks'],dtype=np.float32) for d in detections}
        vec={k:[] for k in ['Global']+([] if probe else ['ROI'])+[f'Learned_{s}' for s in cfg['seeds']]}
        for r in v['rows']:
            arrays=[np.load(ROOT/'features'/(uid+'.npz')) for uid in r['frame_uids']]
            patch=torch.from_numpy(np.stack([z['patch'] for z in arrays])).cuda().float()
            valid=torch.from_numpy(np.stack([z['valid'] for z in arrays])).cuda().float()
            coords=torch.from_numpy(np.stack([z['coords'] for z in arrays])).cuda().float()
            query=torch.from_numpy(np.stack([text['a photo of '+n.lower()] for n in r['objects']]))[None].expand(3,-1,-1).cuda()
            uniform=valid/valid.sum(-1,keepdim=True)
            if not probe:
                rr=np.stack([masks[(uid,tuple(r['objects']))] for uid in r['frame_uids']])
                available=rr.sum(-1)>0
                for frame in range(3):
                    for slot in range(2):
                        if not available[frame,slot]:rr[frame,slot]=arrays[frame]['valid']
                rr=rr/rr.sum(-1,keepdims=True);roi=torch.from_numpy(rr).cuda()
            with torch.no_grad():
                vec['Global'].append(pool_vector(patch,uniform[:,None,:].expand(-1,2,-1),coords).mean(0).cpu().numpy())
                if not probe:vec['ROI'].append(pool_vector(patch,roi,coords).mean(0).cpu().numpy())
                for seed,m in zip(cfg['seeds'],models):
                    out=m(patch,query,valid,coords)[0];vec[f'Learned_{seed}'].append(out.mean(0).cpu().numpy())
            coverage.append({'id':r['id'],'task':r['task'],'split':r['split'],'kind':r['kind'],
                'available_slots':None if probe else int(available.sum()),'total_slots':6,'objects':r['objects']})
        np.savez(path,ids=np.array([r['id'] for r in v['rows']]),**{k:np.stack(x) for k,x in vec.items()})
        dump(dest/(v['video']+'.json'),coverage[-len(v['rows']):])
        if (iv+1)%20==0:print('EMBEDDINGS',iv+1,len(videos),flush=True)

def evaluate(probe=False):
    torch.set_num_threads(4);cfg=read(ROOT/'execution.json');started=time.monotonic()
    embeddings(probe);videos=read(ROOT/'feature-index.json');allresults={};boots={};delta={};scores_all=[]
    for task in TASKS:
        rows=[];vectors=collections.defaultdict(list)
        for v in videos:
            if v['task']!=task:continue
            z=np.load(ROOT/'embeddings'/(v['video']+'.npz'));assert z['ids'].tolist()==[r['id'] for r in v['rows']]
            rows.extend(v['rows'])
            for k in z.files:
                if k!='ids':vectors[k].append(z[k])
        old={r['id']:r for r in read(OEV/'results'/task/'scores.json')}
        assert all(r['id'] in old and r['kind']==old[r['id']]['kind'] and r['split']==old[r['id']]['split'] for r in rows)
        scores={k:np.array([old[r['id']]['scores'][k] for r in rows]) for k in ['AEM','Duration','AEM+O1','AEM+O3']}
        scores['AEM+Duration']=(scores['AEM']+scores['Duration'])/2
        raw={'AEM':np.array([old[r['id']]['raw']['AEM'] for r in rows])};fits={}
        for name,x in vectors.items():
            raw[name],scores[name],fits[name]=density(np.concatenate(x),rows)
            scores['AEM+'+name]=(scores['AEM']+scores[name])/2
        val=np.array([r['split']=='val' for r in rows]);test=np.array([r['split']=='test' for r in rows])
        assert all(r['kind']==0 for r in rows if r['split'] in ['train','val'])
        yy=np.array([r['kind']>0 for r in rows])[test];step=np.array([r['step'] for r in rows])[test]
        vv=np.array([r['video'] for r in rows])[test];uni=np.unique(vv)
        eligible=[int(a) for a in sorted(set(step)) if len(set(yy[step==a]))==2]
        assert eligible==({'quesadilla':[1,6,7,8],'pinwheels':[1,2,3,4,5,6,7,11,12]}[task]),eligible
        thresholds={k:float(np.quantile(s[val],.9,method='higher')) for k,s in scores.items()}
        names=list(scores);res={};by_step={};threshold_stats={}
        for name in names:
            s=scores[name][test];values=[];by_step[name]={}
            for a in eligible:
                mask=step==a;roc=roc_auc_score(yy[mask],s[mask]);check=auc(yy[mask],s[mask],np.ones(mask.sum()))
                assert abs(roc-check)<1e-12
                by_step[name][str(a)]={'auc':roc,'ap':average_precision_score(yy[mask],s[mask]),
                    'errors':int(yy[mask].sum()),'normals':int((~yy[mask]).sum()),'videos':len(np.unique(vv[mask]))}
                values.append(roc)
            res[name]=float(np.mean(values));alarm=s>=thresholds[name]
            threshold_stats[name]={'threshold':thresholds[name],'recall':float(alarm[yy].mean()),'fpr':float(alarm[~yy].mean())}
        learned=[f'AEM+Learned_{seed}' for seed in cfg['seeds']]
        primary=float(np.mean([res[k] for k in learned]));res['AEM+Learned_seed_mean']=primary
        threshold_stats['AEM+Learned_seed_mean']={k:float(np.mean([threshold_stats[m][k] for m in learned])) for k in ['recall','fpr']}
        rng=np.random.default_rng(20261027+TASKS.index(task));draws={k:[] for k in ['aem','learned','gain']+([] if probe else ['roi','observer_gain'])}
        for b in range(cfg['bootstrap']):
            counts=np.bincount(rng.integers(0,len(uni),len(uni)),minlength=len(uni));weight=counts[np.searchsorted(uni,vv)]
            means={}
            for name in ['AEM']+([] if probe else ['AEM+ROI'])+learned:
                vals=[auc(yy[step==a],scores[name][test][step==a],weight[step==a]) for a in eligible]
                means[name]=float(np.mean(vals))
            lm=float(np.mean([means[k] for k in learned]));a=means['AEM']
            items=[('aem',a),('learned',lm),('gain',lm-a)]
            if not probe:items += [('roi',means['AEM+ROI']),('observer_gain',lm-means['AEM+ROI'])]
            for k,x in items:draws[k].append(x)
        def ci(x):
            x=np.array(x);return np.quantile(x[np.isfinite(x)],[.025,.975]).tolist()
        allresults[task]={'macro_auc':res,'by_step':by_step,'working_points':threshold_stats,'eligible_steps':eligible,
            'n_test':int(test.sum()),'errors':int(yy.sum()),'normals':int((~yy).sum()),'videos':len(uni),
            'gain_over_aem':{'point':primary-res['AEM'],'ci95':ci(draws['gain'])},
            'gain_over_roi':None if probe else {'point':primary-res['AEM+ROI'],'ci95':ci(draws['observer_gain'])}}
        boots[task]={k:np.array(x) for k,x in draws.items()}
        dump(ROOT/'results'/task/'fits.json',finite(fits))
        dump(ROOT/'results'/task/'results.json',finite(allresults[task]))
        records=[dict(r,raw={k:float(s[i]) for k,s in raw.items()},scores={k:float(s[i]) for k,s in scores.items()}) for i,r in enumerate(rows)]
        dump(ROOT/'results'/task/'scores.json',records);scores_all.extend(records)
        np.savez(ROOT/'results'/task/'bootstrap.npz',**boots[task])
        print('TASK_SCORED',task,json.dumps(res),flush=True)
    overall={}
    for key in ['gain']+([] if probe else ['observer_gain']):
        x=np.mean([boots[t][key] for t in TASKS],axis=0);good=np.isfinite(x)
        points=[allresults[t]['gain_over_aem' if key=='gain' else 'gain_over_roi']['point'] for t in TASKS]
        overall[key]={'point':float(np.mean(points)),'ci95':np.quantile(x[good],[.025,.975]).tolist(),'valid_bootstrap':int(good.sum())}
    wp={t:{k:allresults[t]['working_points']['AEM+Learned_seed_mean'][k]-allresults[t]['working_points']['AEM'][k]
           for k in ['recall','fpr']} for t in TASKS}
    gate={'gain_ge_3_points':overall['gain']['point']>=.03,'gain_ci_positive':overall['gain']['ci95'][0]>0,
        'both_tasks_positive':all(allresults[t]['gain_over_aem']['point']>0 for t in TASKS)}
    aem_passed=all(gate.values())
    if not probe:gate['learned_over_roi_ci_positive']=overall['observer_gain']['ci95'][0]>0
    passed=all(gate.values()) and not probe
    report={'tasks':allresults,'overall':overall,'gate':gate,'gate_passed':passed,'working_point_deltas':wp,
        'aem_gate_passed':aem_passed,'probe_only':probe,
        'status':'AEM_probe_passed_pending_ROI' if probe and aem_passed else 'completed_G1_early_stop' if probe else 'completed_G1',
        'next':'run_matched_ROI_comparison' if probe and aem_passed else 'review working-point and mechanism evidence before G3' if passed else 'stop_current_CAVE_G1_no_CEI_or_CC4D',
        'configuration_sha256':sha(ROOT/'execution.json'),'source_sha256':sha(__file__),'seconds':time.monotonic()-started,
        'cautions':['GT segments and intent; same-step macro AUROC, not official frame AUC',
                    'Quesadilla and Pinwheels are previously explored development tasks',
                    'bootstrap conditional on fitted normal reference and the user-requested single trained observer',
                    'normal calibration is empirical; no finite-sample conformal FPR claim',
                    'normal-validation supervision selected checkpoint; separate error-free calibration data not added']}
    dump(ROOT/'summary.json',finite(report))
    if probe:dump(ROOT/'probe-summary.json',finite(report))
    dump(ROOT/'status.json',{'phase':report['status'],'gate_passed':passed,'next':report['next']})
    print('EVALUATION_COMPLETE',json.dumps(finite({'overall':overall,'gate':gate,'working_point_deltas':wp})),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');evaluate(ap.parse_args().probe)
