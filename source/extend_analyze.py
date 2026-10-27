"""Five-task summary, with the three new tasks as the primary population."""
import hashlib,json,time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import ROOT,read,dump,sha
from evaluate import finite
from confirm_analysis import Curve,METRICS,metric_dict,interval

EXT=ROOT/'remaining-tasks'
PAIRS={'primary':('AEM+Duration+ROI','AEM+Duration'),
       'learned_vs_roi':('AEM+Duration+Learned','AEM+Duration+ROI'),
       'roi_vs_global':('AEM+Duration+ROI','AEM+Duration+Global'),
       'over_aem':('AEM+Duration+ROI','AEM-raw')}
NAMES=['AEM-raw','AEM','Duration','Global','ROI','Learned','AEM+Duration','AEM+Global','AEM+ROI','AEM+Learned',
       'AEM+Duration+Global','AEM+Duration+ROI','AEM+Duration+Learned']

def main():
    cfg=read(EXT/'config.json');prior=read(ROOT/'follow-up/results.json');frozen=read(EXT/'frozen-inputs.json')
    assert sha(EXT/'config.json')==frozen['config_sha256']
    for p,h in frozen['original'].items():assert sha(ROOT/p)==h,p
    cache=EXT/'metric-cache';cache.mkdir(exist_ok=True);allres={};allboots={};maxerr=0;start=time.monotonic()
    for task in cfg['all_tasks']:
        isnew=task in cfg['tasks']
        rows=read((EXT/'results' if isnew else ROOT/'follow-up/analysis')/task/'scores.json')
        scores={n:np.array([r['scores'][n] for r in rows]) for n in NAMES if n in rows[0]['scores']}
        original=read(ROOT.parent/'aem-duration-complementarity/aem'/task/'scores.json')
        raw_aem={r['id']:r['aem_raw'] for r in original}
        scores['AEM-raw']=np.array([raw_aem[r['id']] for r in rows])
        scores['AEM+Duration+Global']=(scores['AEM']+scores['Duration']+scores['Global'])/3
        assert set(NAMES)==set(scores)
        test=np.array([r['split']=='test' for r in rows]);val=np.array([r['split']=='val' for r in rows]);tt=[r for r in rows if r['split']=='test']
        y=np.array([r['kind']>0 for r in tt]);step=np.array([r['step'] for r in tt]);vid=np.array([r['video'] for r in tt]);uv=np.unique(vid)
        eligible=cfg['eligible_steps'][task] if isnew else prior['tasks'][task]['eligible_steps']
        assert eligible==[int(a) for a in sorted(set(step)) if min(y[step==a].sum(),(~y[step==a]).sum())>=2]
        masks=[step==a for a in eligible];supported=np.array([y[m].sum()>=8 for m in masks]);supported_steps=[a for a,s in zip(eligible,supported) if s]
        offset=10+cfg['tasks'].index(task) if isnew else ['quesadilla','pinwheels'].index(task)
        rng=np.random.default_rng(cfg['seed']+offset);weights=[]
        for _ in range(cfg['bootstrap']):
            counts=np.bincount(rng.integers(0,len(uv),len(uv)),minlength=len(uv));weights.append(counts[np.searchsorted(uv,vid)])
        weights=np.stack(weights);methods={};boots={};sboots={};alarms={}
        for name in NAMES:
            ss=scores[name];s=ss[test];assert np.isfinite(ss).all()
            curves=[Curve(y[m],s[m]) for m in masks];pooled=Curve(y,s)
            bp=np.stack([c(np.ones(m.sum())) for c,m in zip(curves,masks)]);point=np.stack([bp.mean(0),pooled(np.ones(len(y)))])
            path=cache/(task+'-'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.npz')
            sig=hashlib.sha256(s.tobytes()+y.tobytes()+weights.tobytes()+sha(EXT/'config.json').encode()).hexdigest()
            supported_draw=np.full(cfg['bootstrap'],np.nan)
            oldpath=ROOT/'follow-up/metric-cache'/(task+'-'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.npz')
            if path.exists():
                z=np.load(path);assert str(z['signature'])==sig;draw=z['draw'];supported_draw=z['supported_auc']
            elif not isnew and name in prior['tasks'][task]['methods']:
                draw=np.load(oldpath)['draw']
                # Reuse validated old-task results and draws; no old-task rerun.
                for j,scope in enumerate(['macro','pooled']):
                    assert max(abs(point[j,k]-prior['tasks'][task]['methods'][name][scope][m]['point']) for k,m in enumerate(METRICS))<1e-12
                np.savez_compressed(path,signature=sig,draw=draw,supported_auc=supported_draw)
            else:
                draw=np.empty((cfg['bootstrap'],2,len(METRICS)))
                for b,w in enumerate(weights):
                    vals=np.stack([c(w[m]) for c,m in zip(curves,masks)])
                    draw[b,0]=vals.mean(0);draw[b,1]=pooled(w)
                    if isnew:supported_draw[b]=vals[supported,0].mean()
                np.savez_compressed(path,signature=sig,draw=draw,supported_auc=supported_draw)
            for m in masks+[np.ones(len(y),bool)]:
                got=Curve(y[m],s[m])(np.ones(m.sum()))
                err=max(abs(got[0]-roc_auc_score(y[m],s[m])),abs(got[1]-average_precision_score(y[m],s[m])))
                assert err<1e-12;maxerr=max(maxerr,float(err))
            th=float(np.quantile(ss[val],.9,method='higher'));alarm=s>=th;alarms[name]=alarm
            deploy={'threshold':th,'recall':float(alarm[y].mean()),'fpr':float(alarm[~y].mean())}
            deploy['by_kind']={str(k):{'n':sum(r['kind']==k for r in tt),'recall':float(alarm[np.array([r['kind']==k for r in tt])].mean())}
                               for k in [1,2] if any(r['kind']==k for r in tt)}
            methods[name]={'macro':metric_dict(point[0],draw[:,0]),'pooled':metric_dict(point[1],draw[:,1]),
                'by_step':{str(a):dict(zip(METRICS,bp[j].tolist()))|{'errors':int(y[masks[j]].sum()),'normals':int((~y[masks[j]]).sum()),
                    'intended':next(r['intended'] for r in tt if r['step']==a)} for j,a in enumerate(eligible)},
                'deployment':deploy,'supported_macro_auc':{'point':float(bp[supported,0].mean()),**interval(supported_draw)} if isnew else None}
            boots[name]=draw;sboots[name]=supported_draw
            print('EXT_METRIC',task,name,round(point[0,0]*100,4),flush=True)
        contrasts={}
        for key,(a,b) in PAIRS.items():
            contrast={'methods':[a,b]}
            for j,scope in enumerate(['macro','pooled']):
                pp=np.array([methods[a][scope][m]['point']-methods[b][scope][m]['point'] for m in METRICS])
                contrast[scope]=metric_dict(pp,boots[a][:,j]-boots[b][:,j])
            if isnew:
                contrast['supported_macro_auc']={'point':methods[a]['supported_macro_auc']['point']-methods[b]['supported_macro_auc']['point'],**interval(sboots[a]-sboots[b])}
            contrasts[key]=contrast
        a,b=PAIRS['primary'];events={}
        for label,m in [('new_TP',alarms[a]&~alarms[b]&y),('lost_TP',~alarms[a]&alarms[b]&y),('new_FP',alarms[a]&~alarms[b]&~y),('resolved_FP',~alarms[a]&alarms[b]&~y)]:
            selected=[{k:r[k] for k in ['id','video','step','kind','intended']} for r,yes in zip(tt,m) if yes]
            events[label]={'segments':len(selected),'videos':len({r['video'] for r in selected}),'records':selected}
        dump(EXT/'analysis'/task/'events.json',events)
        allres[task]={'methods':methods,'contrasts':contrasts,'segments':len(tt),'errors':int(y.sum()),'videos':len(uv),
                      'eligible_steps':eligible,'supported_steps':supported_steps,'new_task_for_this_observer':isnew}
        allboots[task]=boots
        dump(EXT/'analysis'/task/'results.json',finite(allres[task]))
    aggregates={}
    for group,tasks in [('new_three',cfg['tasks']),('prior_two',['quesadilla','pinwheels']),('all_five',cfg['all_tasks'])]:
        methods={};contrasts={}
        for name in NAMES:
            methods[name]={}
            for j,scope in enumerate(['macro','pooled']):
                point=np.mean([[allres[t]['methods'][name][scope][m]['point'] for m in METRICS] for t in tasks],axis=0)
                draw=np.mean([allboots[t][name][:,j] for t in tasks],axis=0);methods[name][scope]=metric_dict(point,draw)
        for key,(a,b) in PAIRS.items():
            contrasts[key]={'methods':[a,b]}
            for j,scope in enumerate(['macro','pooled']):
                point=np.array([methods[a][scope][m]['point']-methods[b][scope][m]['point'] for m in METRICS])
                draw=np.mean([allboots[t][a][:,j]-allboots[t][b][:,j] for t in tasks],axis=0)
                contrasts[key][scope]=metric_dict(point,draw)
            if group=='new_three':
                point=np.mean([allres[t]['contrasts'][key]['supported_macro_auc']['point'] for t in tasks]);dd=[]
                for t in tasks:
                    dd.append(np.load(cache/(t+'-'+hashlib.sha256(a.encode()).hexdigest()[:12]+'.npz'))['supported_auc']-
                              np.load(cache/(t+'-'+hashlib.sha256(b.encode()).hexdigest()[:12]+'.npz'))['supported_auc'])
                contrasts[key]['supported_macro_auc']={'point':float(point),**interval(np.mean(dd,axis=0))}
        aggregates[group]={'tasks':tasks,'methods':methods,'contrasts':contrasts}
    primary=aggregates['new_three']['contrasts']['primary']['macro']['auc']
    gate={'new_three_primary_ci_lower_positive':primary['ci95'][0]>0,
          'all_three_primary_points_positive':all(allres[t]['contrasts']['primary']['macro']['auc']['point']>0 for t in cfg['tasks'])}
    result={'tasks':allres,'aggregates':aggregates,'gate':gate,'gate_passed':all(gate.values()),'new_training_runs':0,
        'config_sha256':sha(EXT/'config.json'),'source_sha256':sha(__file__),'seconds':time.monotonic()-start,
        'verification':{'old_files_unchanged':True,'max_sklearn_auc_ap_error':maxerr,'old_two_bootstrap_reused':True}}
    dump(EXT/'summary.json',finite(result))
    print('FIVE_TASK_SUMMARY',json.dumps(finite({'primary':primary,'gate':gate})),flush=True)

if __name__=='__main__':main()
