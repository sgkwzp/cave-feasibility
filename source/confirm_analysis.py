"""Frozen, paired confirmation using old scores and separately computed ROI scores."""
import argparse, hashlib, json, time
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
from common import ROOT,TASKS,read,dump,sha
from evaluate import finite

OUT=ROOT/'follow-up'
METRICS=['auc','ap','pauc_0_10','tpr_interp_5','tpr_interp_10','tpr_interp_20',
         'tpr_deterministic_5','tpr_deterministic_10','tpr_deterministic_20']
PAIRS={'utility':('AEM+Duration+Learned','AEM+Duration'),
       'over_aem':('AEM+Learned','AEM'),
       'observer':('AEM+Learned','AEM+ROI'),
       'observer_with_duration':('AEM+Duration+Learned','AEM+Duration+ROI'),
       'learned_vs_global':('AEM+Learned','AEM+Global')}

class Curve:
    def __init__(self,y,s):
        self.order=np.argsort(-s,kind='stable');self.y=y[self.order]
        self.s=s[self.order];self.starts=np.r_[0,np.flatnonzero(np.diff(self.s)!=0)+1]
    def __call__(self,w):
        w=w[self.order];p=np.add.reduceat(w*self.y,self.starts);n=np.add.reduceat(w*~self.y,self.starts)
        keep=p+n>0;p=p[keep];n=n[keep]
        if p.sum()==0 or n.sum()==0:return np.full(len(METRICS),np.nan)
        tp=np.cumsum(p);fp=np.cumsum(n);t=np.r_[0,tp/tp[-1]];f=np.r_[0,fp/fp[-1]]
        a=float(np.trapz(t,f));ap=float((p/tp[-1]*tp/(tp+fp)).sum())
        m=f<=.1;px=np.r_[f[m],.1];py=np.r_[t[m],np.interp(.1,f,t)]
        pa=float(np.trapz(py,px)/.1);budgets=[.05,.1,.2]
        return np.array([a,ap,pa]+[np.interp(b,f,t) for b in budgets]+[t[f<=b].max() for b in budgets])

def interval(x):
    x=np.asarray(x);m=np.isfinite(x)
    return {'ci95':np.quantile(x[m],[.025,.975]).tolist() if m.any() else None,'valid_bootstrap':int(m.sum())}

def metric_dict(point,draw):
    return {k:{'point':float(point[i]),**interval(draw[:,i])} for i,k in enumerate(METRICS)}

def score_records(task,roi):
    rows=read(ROOT/'results'/task/'scores.json')
    extra={r['id']:r for r in read(OUT/'roi-results'/task/'scores.json')} if roi else {}
    scores={k:np.array([r['scores'][k] for r in rows]) for k in ['AEM','Duration','AEM+Duration','Global','AEM+Global']}
    scores['Learned']=np.array([r['scores']['Learned_20261027'] for r in rows])
    scores['AEM+Learned']=(scores['AEM']+scores['Learned'])/2
    scores['AEM+Duration+Learned']=(scores['AEM']+scores['Duration']+scores['Learned'])/3
    if roi:
        assert set(extra)=={r['id'] for r in rows}
        for k in ['ROI','ROI-centroid','Learned-centroid']:
            scores[k]=np.array([extra[r['id']]['scores'][k] for r in rows])
            scores['AEM+'+k]=(scores['AEM']+scores[k])/2
        scores['AEM+Duration+ROI']=(scores['AEM']+scores['Duration']+scores['ROI'])/3
    assert all(np.isfinite(s).all() for s in scores.values())
    return rows,scores

def main(roi):
    cfg=read(OUT/'config.json');start=time.monotonic();allres={};allboots={};allpoints={};maxerr=0
    frozen=read(OUT/'frozen-inputs.json')
    for p,h in frozen['original'].items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'config.json')==frozen['config_sha256']
    cache=OUT/'metric-cache';cache.mkdir(exist_ok=True)
    for ti,task in enumerate(TASKS):
        rows,scores=score_records(task,roi);test=np.array([r['split']=='test' for r in rows]);val=np.array([r['split']=='val' for r in rows])
        assert all(r['kind']==0 for r in rows if r['split']!='test')
        rr=[r for r in rows if r['split']=='test'];y=np.array([r['kind']>0 for r in rr]);steps=np.array([r['step'] for r in rr]);vid=np.array([r['video'] for r in rr]);uv=np.unique(vid)
        eligible=cfg['eligible_steps'][task];masks=[steps==a for a in eligible]
        assert all(len(np.unique(y[m]))==2 for m in masks)
        rng=np.random.default_rng(cfg['seed']+ti);weights=[]
        for _ in range(cfg['bootstrap']):
            count=np.bincount(rng.integers(0,len(uv),len(uv)),minlength=len(uv));weights.append(count[np.searchsorted(uv,vid)])
        weights=np.stack(weights);methods={};boots={};points={};alarms={};checks=[]
        for name,ss in scores.items():
            s=ss[test];curves=[Curve(y[m],s[m]) for m in masks];pooled=Curve(y,s)
            bp=np.stack([c(np.ones(m.sum())) for c,m in zip(curves,masks)]);point=np.stack([bp.mean(0),pooled(np.ones(len(y)))])
            sig=hashlib.sha256(name.encode()+s.tobytes()+y.tobytes()+weights.tobytes()+sha(OUT/'config.json').encode()).hexdigest()
            path=cache/(task+'-'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.npz')
            if path.exists():
                z=np.load(path);assert str(z['signature'])==sig;draw=z['draw']
            else:
                draw=np.empty((cfg['bootstrap'],2,len(METRICS)))
                for b,w in enumerate(weights):
                    draw[b,0]=np.mean([c(w[m]) for c,m in zip(curves,masks)],axis=0);draw[b,1]=pooled(w)
                np.savez_compressed(path,signature=sig,draw=draw)
            # Independent sklearn reconstruction on actual full and weighted data, including ties.
            for m in masks+[np.ones(len(y),bool)]:
                for w in [np.ones(m.sum()),weights[0,m]]:
                    if len(np.unique(y[m][w>0]))<2:continue
                    got=Curve(y[m],s[m])(w);f,t,_=roc_curve(y[m],s[m],sample_weight=w,drop_intermediate=False)
                    std=roc_auc_score(y[m],s[m],sample_weight=w,max_fpr=.1)
                    partial=(2*(std-.5)*(.1-.005)+.005)/.1
                    ref=np.array([roc_auc_score(y[m],s[m],sample_weight=w),average_precision_score(y[m],s[m],sample_weight=w),partial]+
                                 [np.interp(b,f,t) for b in [.05,.1,.2]]+[t[f<=b].max() for b in [.05,.1,.2]])
                    err=float(np.max(np.abs(got-ref)));assert err<1e-10,(name,err);maxerr=max(maxerr,err)
            threshold=float(np.quantile(ss[val],.9,method='higher'));alarm=s>=threshold;alarms[name]=alarm
            deploy=np.array([alarm[y].mean(),alarm[~y].mean()]);depdraw=np.column_stack([(weights*(alarm&y)).sum(1)/(weights*y).sum(1),
                                  (weights*(alarm&~y)).sum(1)/(weights*~y).sum(1)])
            boots[name]=draw;points[name]=point
            bykind={str(k):{'errors':sum(r['kind']==k for r in rr),'recall':float(alarm[np.array([r['kind']==k for r in rr])].mean())}
                    for k in sorted({r['kind'] for r in rr if r['kind']>0})}
            methods[name]={'macro':metric_dict(point[0],draw[:,0]),'pooled':metric_dict(point[1],draw[:,1]),
                'by_step':{str(a):dict(zip(METRICS,bp[j].tolist()))|{'errors':int(y[masks[j]].sum()),'normals':int((~y[masks[j]]).sum())} for j,a in enumerate(eligible)},
                'deployment':{'threshold':threshold,'recall':{'point':deploy[0],**interval(depdraw[:,0])},'fpr':{'point':deploy[1],**interval(depdraw[:,1])},'by_kind':bykind}}
            print('METRIC',task,name,'macro_auc',round(point[0,0]*100,4),flush=True)
        contrasts={};eventlists={}
        for key,(a,b) in PAIRS.items():
            if a not in scores or b not in scores:continue
            contrasts[key]={'methods':[a,b],**{scope:metric_dict(points[a][j]-points[b][j],boots[a][:,j]-boots[b][:,j]) for j,scope in enumerate(['macro','pooled'])}}
            eventlists[key]={}
            for label,mask in [('new_TP',alarms[a]&~alarms[b]&y),('lost_TP',~alarms[a]&alarms[b]&y),('new_FP',alarms[a]&~alarms[b]&~y),('resolved_FP',~alarms[a]&alarms[b]&~y)]:
                selected=[{k:r[k] for k in ['id','video','step','kind','intended']} for r,m in zip(rr,mask) if m]
                eventlists[key][label]={'segments':len(selected),'videos':len({r['video'] for r in selected}),'records':selected}
        records=[{k:r[k] for k in ['id','task','video','split','step','kind','intended']}|{'scores':{k:float(s[i]) for k,s in scores.items()}} for i,r in enumerate(rows)]
        dump(OUT/'analysis'/task/'scores.json',records);dump(OUT/'analysis'/task/'events.json',eventlists)
        allres[task]={'methods':methods,'contrasts':contrasts,'segments':len(rr),'errors':int(y.sum()),'videos':len(uv),'eligible_steps':eligible}
        allboots[task]=boots;allpoints[task]=points
    names=list(allpoints[TASKS[0]]);overall={};contrasts={}
    for name in names:
        pp=np.mean([allpoints[t][name] for t in TASKS],axis=0);dd=np.mean([allboots[t][name] for t in TASKS],axis=0)
        overall[name]={scope:metric_dict(pp[j],dd[:,j]) for j,scope in enumerate(['macro','pooled'])}
    for key,(a,b) in PAIRS.items():
        if a not in names or b not in names:continue
        pp=np.mean([allpoints[t][a]-allpoints[t][b] for t in TASKS],axis=0)
        dd=np.mean([allboots[t][a]-allboots[t][b] for t in TASKS],axis=0)
        contrasts[key]={'methods':[a,b],**{scope:metric_dict(pp[j],dd[:,j]) for j,scope in enumerate(['macro','pooled'])}}
    utility=contrasts['utility']['macro'];gate={
        'utility_macro_auc_ci_lower_positive':utility['auc']['ci95'][0]>0,
        'utility_macro_auc_both_task_points_positive':all(allres[t]['contrasts']['utility']['macro']['auc']['point']>0 for t in TASKS),
        'utility_macro_pauc_mean_nonnegative':utility['pauc_0_10']['point']>=0,
        'observer_macro_auc_ci_lower_positive':contrasts['observer']['macro']['auc']['ci95'][0]>0 if roi else None}
    result={'tasks':allres,'overall':overall,'contrasts':contrasts,'gate':gate,'gate_passed':roi and all(gate.values()),
            'roi_included':roi,'seconds':time.monotonic()-start,'config_sha256':sha(OUT/'config.json'),'source_sha256':sha(__file__),
            'verification':{'original_files_unchanged':True,'max_sklearn_metric_error':maxerr}}
    dump(OUT/('results.json' if roi else 'utility-results.json'),finite(result))
    print('COMPLETE',json.dumps(finite({'contrasts':{k:v['macro']['auc'] for k,v in contrasts.items()},'gate':gate})),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--roi',action='store_true');main(ap.parse_args().roi)
