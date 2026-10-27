"""Freeze two bounded follow-ups; predictions do not access test intent/errors."""
import collections,datetime
import numpy as np
from common import ROOT,PROJECT,read,dump,sha

OUT=ROOT/'deployment-mechanism'
TASKS=['coffee','pinwheels','oatmeal','quesadilla','tea']
SEG=ROOT.parent/'aem-segmentation-shift'

def main():
    cfg={'tasks':TASKS,'seed':20261027,'bootstrap':2000,'pre_offsets':[-.6,-.4,-.2],
         'tail_fractions':[.85,.90,.95],'training_runs':0,'eligible_steps':{
         'coffee':[1,2,3,4,6,7,8,9,10,11,12,13,14,15],'pinwheels':[1,2,3,4,5,6,7,11,12],
         'oatmeal':[1,3,8,9,11,15],'quesadilla':[1,6,7,8],'tea':[2,3,5,8,9]}}
    dump(OUT/'config.json',cfg)
    queries=read(ROOT/'queries.json')|read(ROOT/'remaining-tasks/queries.json')
    old=read(ROOT/'manifest.json')+read(ROOT/'remaining-tasks/manifest.json')
    byvideo={v['video']:v for v in old};videos=[];sources={};counts={}
    for task in TASKS:
        counts[task]=collections.Counter()
        for split in ['train','val','test']:
            for p in sorted((SEG/'cache/scores'/task/split).glob('*.npz')):
                z=np.load(p);v=byvideo[p.stem];assert v['split']==split
                rows=[]
                for r in v['rows']:
                    row={k:r[k] for k in ['id','task','video','segment','step','kind','split','start','end','duration','intended']}
                    row.update(arm='Pre',source_id=r['id'],id='Pre:'+r['id'],objects=queries[f"{task}:{r['step']}"],
                               times=[max(0.,r['start']+offset) for offset in cfg['pre_offsets']],upper=None)
                    rows.append(row)
                for i,(span,a,sim) in enumerate(zip(z['p_spans'],z['p_labels'],z['p_sims'])):
                    if a==0:continue
                    s,e=[int(x) for x in span];dur=(e-s+1)/10;assert e>=s
                    rows.append({'arm':'Pred','id':f'Pred:{p.stem}:{i}','task':task,'video':p.stem,'segment':i,
                                 'step':int(a),'split':split,'start':s/10,'end':e/10,'duration':dur,
                                 'objects':queries[f'{task}:{a}'],'times':[min(e/10,s/10+f*dur) for f in cfg['tail_fractions']],
                                 'upper':e/10,'aem_raw':float(1-sim)})
                counts[task][split+'_videos']+=1
                for r in rows:counts[task][split+'_'+r['arm']]+=1
                sources[p.relative_to(PROJECT).as_posix()]=sha(p)
                videos.append({'video':p.stem,'task':task,'split':split,'rows':rows,
                               'zip':f'data/data/{task.title()}/{task}_videos.zip','member':f'trim_videos/{p.stem}.mp4'})
    assert len(videos)==389
    assert sum(r['arm']=='Pre' for v in videos for r in v['rows'])==4595
    assert all(r.get('kind',0)==0 for v in videos if v['split']!='test' for r in v['rows'])
    dump(OUT/'manifest.json',videos);dump(OUT/'queries.json',queries)
    for p in [ROOT/'queries.json',ROOT/'remaining-tasks/queries.json',ROOT/'remaining-tasks/report.md',
              ROOT/'source/evaluate.py',ROOT/'source/detect.py',ROOT/'source/observer.py',ROOT/'source/extract.py']:
        sources[p.relative_to(PROJECT).as_posix()]=sha(p)
    dump(OUT/'frozen-inputs.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'files':sources,'protocol_sha256':sha(OUT/'protocol.md'),'config_sha256':sha(OUT/'config.json'),
         'manifest_sha256':sha(OUT/'manifest.json'),'counts':counts})
    dump(OUT/'status.json',{'phase':'prepared','new_training_runs':0});print(counts,flush=True)

if __name__=='__main__':main()
