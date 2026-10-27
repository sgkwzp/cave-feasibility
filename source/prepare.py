"""Freeze old tail windows and align normal-only released teacher records by time."""
import collections
from pathlib import Path
from common import ROOT, PROJECT, OEV, TASKS, read, dump, sha

def main():
    source = read(OEV/'manifest.json')
    data = PROJECT/'code/AEM/data/egoper'
    videos=[]; supervision=[]; queries=collections.defaultdict(list); problems=[]
    provenance={'manifest':sha(OEV/'manifest.json')}
    for task in TASKS:
        dp=data/'detection'/f'{task}_detection.json'
        gp=data/'scene_graph_json'/f'{task}_gpt4o.json'
        det=read(dp)[task]; graphs=read(gp)
        provenance[str(dp.relative_to(PROJECT))]=sha(dp)
        provenance[str(gp.relative_to(PROJECT))]=sha(gp)
        for v0 in source:
            if v0['task']!=task: continue
            v=dict(v0);v['zip']=f'data/{task.title()}/{task}_videos.zip'
            videos.append(v)
            if v['split']=='test':continue
            assert all(r['kind']==0 for r in v['rows'])
            for r in v['rows']:
                matches=[(i,d) for i,d in enumerate(det.get(v['video'],[]))
                    if d.get('img_path') and float(d['time_stamp'][0])>=r['start']-.11
                    and float(d['time_stamp'][1])<=r['end']+.11]
                if not matches:
                    problems.append({'id':r['id'],'reason':'teacher timestamp match','matches':len(matches)})
                    continue
                for di,d in matches:
                    names=[d.get('OBJECT A'),d.get('OBJECT B')]
                    if not all(isinstance(s,str) and s.strip() for s in names):
                        problems.append({'id':r['id'],'reason':'missing object names'});continue
                    key=f"{task}:{r['step']}"
                    if v['split']=='train':queries[key].append(tuple(names))
                    box=[]
                    for slot in ['A','B']:
                        item=d.get('BBox of OBJECT '+slot,{})
                        b=item.get('bbox');p=item.get('prob')
                        box.append(b if b is not None and p is not None and float(p)>=.25 else None)
                    if not any(b is not None for b in box):continue
                    gm=[g for g in graphs.get(v['video'],[]) if g.get('img_path') and
                        g['img_path'].endswith(d['img_path'])]
                    g=gm[0] if len(gm)==1 else {}
                    sec=int(Path(d['img_path']).name.split('_')[0])
                    upper=r['next_action_start'] if r['next_action_start'] is not None else v['object_frames']/10
                    if not r['start']-.2 <= sec < upper:
                        problems.append({'id':r['id'],'reason':'teacher frame outside step plus trailing background','time':sec,'upper':upper})
                        continue
                    supervision.append({'id':r['id']+'#'+str(di),'segment_id':r['id'],'video':v['video'],'task':task,'step':r['step'],
                        'split':r['split'],'names':names,'time':sec,'boxes_cxcywh_normalized':box,
                        'attribute_sentence':g.get('attribute_sentence',[]),'relation_sentence':g.get('relation_sentence',[]),
                        'teacher_image_path':d['img_path'],'image_source':'published effect_frames npz exact img_path',
                        'teacher_subinterval':d['time_stamp']})
    canonical={k:list(collections.Counter(x).most_common(1)[0][0]) for k,x in queries.items()}
    for v in videos:
        for r in v['rows']:
            key=f"{r['task']}:{r['step']}";assert key in canonical,key
            r['objects']=canonical[key]
    counts={t:{s:{'videos':len([v for v in videos if v['task']==t and v['split']==s]),
                  'segments':sum(len(v['rows']) for v in videos if v['task']==t and v['split']==s),
                  'supervision_frames':sum(x['task']==t and x['split']==s for x in supervision)}
               for s in ['train','val','test']} for t in TASKS}
    assert len(videos)==164 and sum(len(v['rows']) for v in videos)==1565
    assert not any(s['split']=='test' for s in supervision)
    dump(ROOT/'manifest.json',videos);dump(ROOT/'supervision.json',supervision)
    dump(ROOT/'queries.json',canonical)
    dump(ROOT/'preparation.json',{'counts':counts,'unavailable_teacher_records':problems,'source_hashes':provenance,
        'teacher_coordinates':'normalized cxcywh; converted through exact letterbox transform',
        'normal_only_teacher':True,'query_names_source':'majority object names in training teacher records per task/step'})
    print(counts,flush=True)
    print('UNMATCHED',len(problems),'SUPERVISION',len(supervision),flush=True)

if __name__=='__main__':main()
