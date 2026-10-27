"""Normal-only reference fits, identical frozen density and fusion rules."""
import json,time
import numpy as np
from common import ROOT,read,dump,sha
from evaluate import density,smooth_score,finite

OUT=ROOT/'deployment-mechanism'

def old_rows(task):
    base=ROOT/'follow-up/analysis' if task in ['quesadilla','pinwheels'] else ROOT/'remaining-tasks/results'
    return read(base/task/'scores.json')

def main():
    cfg=read(OUT/'config.json');start=time.monotonic()
    videos=read(OUT/'manifest.json');meta={v['video']:read(OUT/'observations'/(v['video']+'.json')) for v in videos}
    checks=[]
    for task in cfg['tasks']:
        for arm in ['Pre','Pred']:
            rows=[];vectors=[]
            for v in videos:
                if v['task']!=task:continue
                rr=meta[v['video']]['rows'];z=np.load(OUT/'embeddings'/(v['video']+'.npz'))
                assert z['ids'].tolist()==[r['id'] for r in rr]
                use=[i for i,r in enumerate(rr) if r['arm']==arm]
                rows.extend(rr[i] for i in use);vectors.append(z['ROI'][use])
            raw,score,fit=density(np.concatenate(vectors),rows)
            if arm=='Pre':
                old={r['id']:r for r in old_rows(task)};assert set(old)=={r['source_id'] for r in rows}
                out=[]
                for i,r in enumerate(rows):
                    o=old[r['source_id']];assert all(r[k]==o[k] for k in ['video','step','split','kind'])
                    ss={k:o['scores'][k] for k in ['AEM','Duration','AEM+Duration']}
                    ss.update(Pre=float(score[i]),Post=o['scores']['ROI'],
                              **{'AEM+Duration+Pre':(ss['AEM']+ss['Duration']+float(score[i]))/3,
                                 'AEM+Duration+Post':o['scores']['AEM+Duration+ROI']})
                    assert abs(ss['AEM+Duration+Post']-(ss['AEM']+ss['Duration']+ss['Post'])/3)<1e-12
                    out.append(dict(r,id=r['source_id'],raw_pre=float(raw[i]),scores=ss))
            else:
                steps=np.array([r['step'] for r in rows]);train=np.array([r['split']=='train' for r in rows])
                raws={'AEM':np.array([r['aem_raw'] for r in rows]),'Duration':np.zeros(len(rows)),'ROI':raw}
                scores={'AEM':np.zeros(len(rows)),'Duration':np.zeros(len(rows)),'ROI':score};duration_fit={}
                ld=np.log([r['duration'] for r in rows])
                for a in sorted(set(steps)):
                    m=steps==a;t=m&train;mu=ld[t].mean();sd=max(ld[t].std(ddof=1),1e-6)
                    raws['Duration'][m]=abs((ld[m]-mu)/sd)
                    duration_fit[str(a)]={'n':int(t.sum()),'mean':float(mu),'std':float(sd)}
                    for k in ['AEM','Duration']:scores[k][m]=smooth_score(raws[k][m],raws[k][t])
                scores['AEM-raw']=raws['AEM'].copy()
                scores['AEM+Duration']=(scores['AEM']+scores['Duration'])/2
                scores['AEM+ROI']=(scores['AEM']+scores['ROI'])/2
                scores['AEM+Duration+ROI']=(scores['AEM']+scores['Duration']+scores['ROI'])/3
                assert all(np.isfinite(s).all() for s in scores.values())
                out=[dict(r,raw={k:float(s[i]) for k,s in raws.items()},scores={k:float(s[i]) for k,s in scores.items()}) for i,r in enumerate(rows)]
                fit['duration']=duration_fit
            dump(OUT/'scores'/arm/task/'scores.json',out);dump(OUT/'scores'/arm/task/'fits.json',finite(fit))
            checks.append({'task':task,'arm':arm,'segments':len(rows),'train':sum(r['split']=='train' for r in rows)})
            print('SCORED',task,arm,len(rows),flush=True)
    dump(OUT/'score-verification.json',{'checks':checks,'max_anchor_error':max(m['anchor_max_error'] for m in meta.values()),
         'source_sha256':sha(__file__),'density_source_sha256':sha(ROOT/'source/evaluate.py'),'seconds':time.monotonic()-start})
    dump(OUT/'status.json',{'phase':'scores_complete'})

if __name__=='__main__':main()
