"""Freeze remaining-task inputs, normal-only object queries and matched baselines."""
import collections,datetime
import numpy as np
from common import ROOT,PROJECT,read,dump,sha
from evaluate import smooth_score

EXT=ROOT/'remaining-tasks';OLD=ROOT.parent/'aem-duration-complementarity'

def main():
    cfg=read(EXT/'config.json');source=read(OLD/'manifest.json');ann=read(PROJECT/'code/EgoPER/data/annotation.json')
    videos=[];queries={};counts={};baseline_checks={};provenance={};sigs={}
    for task in cfg['all_tasks']:
        arr=read(OLD/'aem'/task/'scores.json');lookup={r['id']:r for r in source if r['task']==task and r['eligible']}
        assert set(lookup)=={r['id'] for r in arr}
        rows=[]
        for s in arr:
            r=dict(lookup[s['id']]);assert all(s[k]==r[k] for k in ['video','step','split','start','end'])
            r['aem_raw']=s['aem_raw'];rows.append(r)
        train=np.array([r['split']=='train' for r in rows]);steps=np.array([r['step'] for r in rows])
        assert all(r['kind']==0 for r in rows if r['split']!='test')
        raw={'AEM':np.array([r['aem_raw'] for r in rows]),'Duration':np.zeros(len(rows))};score={k:np.zeros(len(rows)) for k in raw}
        duration=np.log([r['duration'] for r in rows]);fit={}
        for a in sorted(set(steps)):
            m=steps==a;t=m&train;mu=duration[t].mean();sd=max(duration[t].std(ddof=1),1e-6)
            raw['Duration'][m]=abs((duration[m]-mu)/sd);fit[str(a)]={'mean':float(mu),'std':float(sd),'n':int(t.sum())}
            for k in raw:score[k][m]=smooth_score(raw[k][m],raw[k][t])
        for i,r in enumerate(rows):r['raw']={k:float(v[i]) for k,v in raw.items()};r['scores']={k:float(v[i]) for k,v in score.items()}
        if task not in cfg['tasks']:
            old={r['id']:r for r in read(ROOT/'results'/task/'scores.json')}
            err=max(abs(r['scores'][k]-old[r['id']]['scores'][k]) for r in rows for k in score)
            assert err<1e-9,(task,err);baseline_checks[task]=err;continue
        sigs[task]=read(OLD/'aem'/task/'signature.json')
        for name in ['scores.json','signature.json']:provenance[(OLD/'aem'/task/name).relative_to(PROJECT).as_posix()]=sha(OLD/'aem'/task/name)
        detpath=PROJECT/f'code/AEM/data/egoper/detection/{task}_detection.json';det=read(detpath)[task]
        provenance[detpath.relative_to(PROJECT).as_posix()]=sha(detpath)
        qs=collections.defaultdict(list)
        for r in rows:
            if r['split']!='train':continue
            for d in det.get(r['video'],[]):
                if not d.get('img_path'):continue
                if float(d['time_stamp'][0])<r['start']-.11 or float(d['time_stamp'][1])>r['end']+.11:continue
                names=(d.get('OBJECT A'),d.get('OBJECT B'))
                if all(isinstance(n,str) and n.strip() for n in names):qs[r['step']].append(names)
        canonical={a:list(collections.Counter(n).most_common(1)[0][0]) for a,n in qs.items()}
        assert set(steps)==set(canonical),(task,set(steps)-set(canonical));queries.update({f'{task}:{a}':n for a,n in canonical.items()})
        labs={v['video_id']:v['labels'] for v in ann[task]['segments']};groups=collections.defaultdict(list)
        for r in rows:
            lab=labs[r['video']];i=r['segment'];assert lab['action'][i]==r['step'] and lab['action_type'][i]==r['kind']
            nxt=next((float(ts[0]) for a,ts in zip(lab['action'][i+1:],lab['time_stamp'][i+1:]) if a!=0),None)
            r.update(objects=canonical[r['step']],next_action_start=nxt,old_times=[r['start']+f*r['duration'] for f in cfg['tail_fractions']])
            groups[r['video']].append(r)
        for video,rr in groups.items():videos.append({'video':video,'task':task,'split':rr[0]['split'],'rows':rr,
            'zip':f'data/data/{task.title()}/{task}_videos.zip','member':f'trim_videos/{video}.mp4'})
        counts[task]={s:{'videos':len({r['video'] for r in rows if r['split']==s}),'segments':sum(r['split']==s for r in rows)} for s in ['train','val','test']}
        dump(EXT/'baseline'/task/'scores.json',rows);dump(EXT/'baseline'/task/'duration-fit.json',fit)
    dump(EXT/'manifest.json',videos);dump(EXT/'queries.json',queries)
    original=read(ROOT/'follow-up/frozen-inputs.json')['original']
    for p in ['follow-up/results.json','follow-up/secondary-results.json','follow-up/report.md','source/confirm_analysis.py']:
        original[p]=sha(ROOT/p)
    frozen={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original':original,
        'config_sha256':sha(EXT/'config.json'),'protocol_sha256':sha(EXT/'protocol.md'),'manifest_sha256':sha(EXT/'manifest.json'),
        'queries_sha256':sha(EXT/'queries.json'),'source_sha256':sha(__file__),'source_inputs':provenance,'aem_signatures':sigs,
        'counts':counts,'baseline_reconstruction_max_errors':baseline_checks}
    dump(EXT/'frozen-inputs.json',frozen)
    print('PREPARED',counts,'BASELINE_MATCH',baseline_checks,flush=True)

if __name__=='__main__':main()
