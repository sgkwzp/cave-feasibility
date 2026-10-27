"""Paired five-task metrics; all-frame prediction evaluation and matched GT ablation."""
import hashlib,json,sys,time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import ROOT,PROJECT,read,dump,sha
from evaluate import finite
from confirm_analysis import Curve,METRICS,metric_dict,interval

OUT=ROOT/'deployment-mechanism';SEG=ROOT.parent/'aem-segmentation-shift'
PNAMES=['AEM-raw','AEM','Duration','ROI','AEM+Duration','AEM+ROI','AEM+Duration+ROI']
MNAMES=['AEM','Duration','Pre','Post','AEM+Duration','AEM+Duration+Pre','AEM+Duration+Post']
PPAIRS={'primary':('AEM+Duration+ROI','AEM+Duration'),'vs_raw_aem':('AEM+Duration+ROI','AEM-raw'),
        'roi_without_duration':('AEM+ROI','AEM')}
MPAIRS={'primary':('AEM+Duration+Post','AEM+Duration+Pre'),'post_increment':('AEM+Duration+Post','AEM+Duration'),
        'pre_increment':('AEM+Duration+Pre','AEM+Duration'),'standalone':('Post','Pre')}

def item(point,draw):return {'point':float(point),**interval(draw)}

def segmented(rows,scores,eligible,uv,vweights):
    y=np.array([r['kind']>0 for r in rows]);step=np.array([r['step'] for r in rows]);vid=np.array([r['video'] for r in rows])
    w=vweights[:,np.searchsorted(uv,vid)];masks=[step==a for a in eligible]
    assert all(min(y[m].sum(),(~y[m]).sum())>=2 for m in masks)
    keep=np.array([y[m].sum()>=8 for m in masks]);methods={};boots={};maxerr=0.
    for name,s in scores.items():
        cs=[Curve(y[m],s[m]) for m in masks];pool=Curve(y,s)
        bp=np.stack([c(np.ones(m.sum())) for c,m in zip(cs,masks)]);point=np.stack([bp.mean(0),pool(np.ones(len(y)))])
        draws=np.empty((len(w),2,len(METRICS)));supported=np.empty(len(w))
        for b,ww in enumerate(w):
            vals=np.stack([c(ww[m]) for c,m in zip(cs,masks)]);draws[b,0]=vals.mean(0);draws[b,1]=pool(ww)
            supported[b]=vals[keep,0].mean()
        for m,c in zip(masks,cs):
            got=c(np.ones(m.sum()));err=max(abs(got[0]-roc_auc_score(y[m],s[m])),abs(got[1]-average_precision_score(y[m],s[m])))
            assert err<1e-12;maxerr=max(maxerr,err)
        methods[name]={'macro':metric_dict(point[0],draws[:,0]),'pooled':metric_dict(point[1],draws[:,1]),
             'supported':item(bp[keep,0].mean(),supported),'by_step':{
             str(a):{'auc':float(bp[i,0]),'ap':float(bp[i,1]),'errors':int(y[masks[i]].sum()),'normals':int((~y[masks[i]]).sum())} for i,a in enumerate(eligible)}}
        boots[name]={'macro':draws[:,0],'pooled':draws[:,1],'supported':supported}
    return {'methods':methods,'segments':len(rows),'errors':int(y.sum()),'eligible_steps':eligible,'supported_steps':np.array(eligible)[keep].tolist(),
            'max_sklearn_error':float(maxerr)},boots

def contrasts(result,boots,pairs,scopes):
    out={}
    for k,(a,b) in pairs.items():
        out[k]={'methods':[a,b]}
        for scope in scopes:
            if scope=='supported':
                out[k][scope]=item(result['methods'][a][scope]['point']-result['methods'][b][scope]['point'],boots[a][scope]-boots[b][scope])
            else:
                pp=np.array([result['methods'][a][scope][m]['point']-result['methods'][b][scope][m]['point'] for m in METRICS])
                out[k][scope]=metric_dict(pp,boots[a][scope]-boots[b][scope])
    return out

def official(task,rows):
    import torch
    sys.path.insert(0,str(PROJECT/'code/AEM'))
    from metric import evaluate_ed,calculate_micro_auc
    out={};lookup={(r['video'],r['segment']):r for r in rows if r['split']=='test'}
    for name in ['AEM-raw','AEM+Duration','AEM+ROI','AEM+Duration+ROI']:
        records={}
        for p in sorted((SEG/'cache/scores'/task/'test').glob('*.npz')):
            z=np.load(p);sim=z['p_sims'].copy() if name=='AEM-raw' else np.ones(len(z['p_labels']),np.float64)
            if name!='AEM-raw':
                for j,a in enumerate(z['p_labels']):
                    if a>0:sim[j]=np.exp(-lookup[(p.stem,j)]['scores'][name])
            records[p.stem]={'segments':torch.from_numpy(z['p_spans']),'segment_action_pred':torch.from_numpy(z['p_labels']),
                    'segment_action_sim':torch.from_numpy(sim),'frame_error_label':torch.from_numpy(z['gt_error']),
                    'frame_action_label':torch.from_numpy(z['gt_action'])}
        vals=[evaluate_ed(records,threshold=float(t)) for t in np.arange(0,1,.025)]
        out[name]={'auc':float(calculate_micro_auc([v['micro']['fpr'] for v in vals],[v['micro']['tpr'] for v in vals])),
                   'eda':float(np.mean([v['error_acc'] for v in vals]))}
    ref=read(SEG/f'cache/{task}-metadata.json')['splits']['test']['official']
    assert max(abs(100*out['AEM-raw'][k]-ref[k]) for k in ['auc','eda'])<1e-7
    return out

def main():
    cfg=read(OUT/'config.json');start=time.monotonic();allresults={};allboots={};cache=OUT/'metric-cache';cache.mkdir(exist_ok=True)
    for ti,task in enumerate(cfg['tasks']):
        # Existing metric caches are retained only for exactly identical score sources.
        inp={arm:sha(OUT/'scores'/arm/task/'scores.json') for arm in ['Pre','Pred']}
        signature=hashlib.sha256(json.dumps(inp,sort_keys=True).encode()+sha(__file__).encode()+sha(OUT/'config.json').encode()).hexdigest()
        rp=OUT/'analysis'/task/'results.json';bp=cache/(task+'.npz')
        if rp.exists() and bp.exists():
            rr=read(rp)
            if rr['signature']==signature:
                z=np.load(bp);allresults[task]=rr;allboots[task]={arm:{name:{scope:z[f'{arm}|{name}|{scope}'] for scope in (['frame','macro','pooled','supported'] if arm=='P' else ['macro','pooled','supported'])} for name in (PNAMES if arm=='P' else MNAMES)} for arm in ['P','M']};continue
        mr=read(OUT/'scores/Pre'/task/'scores.json');pr=read(OUT/'scores/Pred'/task/'scores.json')
        tt=[r for r in mr if r['split']=='test'];uv=np.unique([r['video'] for r in tt])
        rng=np.random.default_rng(cfg['seed']+ti);vw=np.stack([np.bincount(rng.integers(0,len(uv),len(uv)),minlength=len(uv)) for _ in range(cfg['bootstrap'])])
        ms={k:np.array([r['scores'][k] for r in tt]) for k in MNAMES}
        mm,mb=segmented(tt,ms,cfg['eligible_steps'][task],uv,vw);mm['contrasts']=contrasts(mm,mb,MPAIRS,['macro','pooled','supported'])
        lookup={(r['video'],r['segment']):r for r in pr if r['split']=='test'}
        gtlookup={r['id']:r for r in tt};segment_scores={k:{} for k in PNAMES};record=[];coverage=[];expanded={k:[] for k in PNAMES};expanded_y=[]
        for vi,video in enumerate(uv):
            z=np.load(SEG/'cache/scores'/task/'test'/(video+'.npz'));idx=z['p_index'];assert (idx>=0).all()
            y=z['gt_error']>0;n=len(z['p_labels']);pos=np.bincount(idx,weights=y,minlength=n);neg=np.bincount(idx,weights=~y,minlength=n)
            assert pos.sum()+neg.sum()==len(y);maps={}
            coverage.append({'video':video,'frames':len(y),'error_frames':int(y.sum()),'error_to_bg':int(y[z['pred_action']==0].sum()),
                             'singletons':int((z['p_spans'][:,0]==z['p_spans'][:,1]).sum())})
            for name in PNAMES:
                ss=np.array([0. if a==0 else lookup[(video,j)]['scores'][name] for j,a in enumerate(z['p_labels'])])
                maps[name]=ss;grid=ss[idx];expanded[name].append(grid)
                if name=='AEM-raw':assert np.max(abs(grid-z['PP']))<2e-6
                for j in range(len(z['g_spans'])):
                    key=f'{video}:{j}'
                    if key not in gtlookup:continue
                    mask=z['g_index']==j;assert mask.any(),key
                    segment_scores[name][key]=float(grid[mask].mean())
            expanded_y.append(y)
            record.append({'video':video,'pos':pos,'neg':neg,'scores':maps})
        ss={k:np.array([segment_scores[k][r['id']] for r in tt]) for k in PNAMES}
        pm,pb=segmented(tt,ss,cfg['eligible_steps'][task],uv,vw)
        yy=np.tile([True,False],sum(len(r['pos']) for r in record));basew=np.concatenate([np.stack([r['pos'],r['neg']],1).ravel() for r in record])
        video_index=np.concatenate([np.full(2*len(r['pos']),i) for i,r in enumerate(record)])
        maxerr=0.
        for name in PNAMES:
            s=np.concatenate([np.repeat(r['scores'][name],2) for r in record]);curve=Curve(yy,s)
            point=curve(basew);draw=np.stack([curve(basew*wv[video_index]) for wv in vw])
            actual=np.concatenate(expanded[name]);ey=np.concatenate(expanded_y)
            err=max(abs(point[0]-roc_auc_score(ey,actual)),abs(point[1]-average_precision_score(ey,actual)))
            assert err<1e-12;maxerr=max(maxerr,err)
            pm['methods'][name]['frame']=metric_dict(point,draw);pb[name]['frame']=draw
            val=np.array([r['scores'][name] for r in pr if r['split']=='val']);threshold=float(np.quantile(val,.9,method='higher'))
            alarm=actual>=threshold
            pm['methods'][name]['fixed_threshold_frame']={'threshold':threshold,'tpr':float(alarm[ey].mean()),'fpr':float(alarm[~ey].mean())}
        pm['contrasts']=contrasts(pm,pb,PPAIRS,['frame','macro','pooled','supported']);pm['coverage']=coverage;pm['max_frame_sklearn_error']=float(maxerr)
        pm['official']=official(task,pr)
        rowsout=[{k:r[k] for k in ['id','video','step','kind','split','intended']}|{'scores':{k:float(s[i]) for k,s in ss.items()}} for i,r in enumerate(tt)]
        dump(OUT/'analysis'/task/'predicted_on_gt.json',rowsout)
        rr={'P':pm,'M':mm,'signature':signature};allresults[task]=rr;allboots[task]={'P':pb,'M':mb}
        np.savez_compressed(bp,**{f'{arm}|{name}|{scope}':x for arm,bb in allboots[task].items() for name,values in bb.items() for scope,x in values.items()})
        dump(rp,finite(rr));print('ANALYZED',task,'P_frame',pm['contrasts']['primary']['frame']['auc'],'M_macro',mm['contrasts']['primary']['macro']['auc'],flush=True)
    groups={}
    for group,tasks in [('all_five',cfg['tasks']),('three_identity_clear',['coffee','quesadilla','tea'])]:
        groups[group]={}
        for arm,names,pairs,scopes in [('P',PNAMES,PPAIRS,['frame','macro','pooled','supported']),('M',MNAMES,MPAIRS,['macro','pooled','supported'])]:
            result={'methods':{}};bb={}
            for name in names:
                result['methods'][name]={};bb[name]={}
                for scope in scopes:
                    draw=np.mean([allboots[t][arm][name][scope] for t in tasks],axis=0);bb[name][scope]=draw
                    if scope=='supported':
                        point=np.mean([allresults[t][arm]['methods'][name][scope]['point'] for t in tasks]);result['methods'][name][scope]=item(point,draw)
                    else:
                        point=np.mean([[allresults[t][arm]['methods'][name][scope][m]['point'] for m in METRICS] for t in tasks],axis=0)
                        result['methods'][name][scope]=metric_dict(point,draw)
            result['contrasts']=contrasts(result,bb,pairs,scopes);groups[group][arm]=result
    p=groups['all_five']['P']['contrasts']['primary']['frame']['auc'];m=groups['all_five']['M']['contrasts']['primary']['macro']['auc']
    gates={'P':p['ci95'][0]>0 and sum(allresults[t]['P']['contrasts']['primary']['frame']['auc']['point']>0 for t in cfg['tasks'])>=3,
           'M':m['ci95'][0]>0 and m['point']>=.01}
    summary={'tasks':allresults,'aggregates':groups,'gates':gates,'seconds':time.monotonic()-start,'source_sha256':sha(__file__)}
    dump(OUT/'summary.json',finite(summary));dump(OUT/'status.json',{'phase':'analysis_complete','gates':gates})
    print('VALIDATION_COMPLETE',gates,'P',p,'M',m,flush=True)

if __name__=='__main__':main()
