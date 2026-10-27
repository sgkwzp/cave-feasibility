"""Freeze G2 cohort and assemble observation controls from existing patch caches."""
import collections, hashlib, json, re, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAVE=ROOT.parent
PROJECT=CAVE.parents[2]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def main():
    cfg=read(ROOT/'config.json');t0=time.monotonic()
    videos=read(CAVE/'manifest.json')+read(CAVE/'remaining-tasks/manifest.json')
    allrows=[];missing=[];sources={};repeated=[];potential=collections.Counter()
    for task in cfg['tasks']:
        path=PROJECT/f'code/AEM/data/egoper/scene_graph_json/{task}_gpt4o.json'
        sources[str(path.relative_to(PROJECT))]=sha(path);graphs=read(path)
        paths=collections.defaultdict(list)
        for video,gs in graphs.items():
            for g in gs:
                if g.get('img_path') and (g.get('attribute_sentence') or g.get('relation_sentence')):
                    paths[g['img_path']].append(g)
        repeated.extend([{'task':task,'path':p,'records':len(gs),'identical':len({json.dumps([g.get('attribute_sentence'),g.get('relation_sentence')],sort_keys=True) for g in gs})==1} for p,gs in paths.items() if len(gs)>1])
        for v in videos:
            if v['task']!=task or v['split']=='test':continue
            for r in v['rows']:
                assert r['kind']==0
                potential[task+':'+v['split']]+=1
                gg=[g for g in graphs.get(v['video'],[]) if g.get('attribute_sentence') and g.get('relation_sentence') and float(g['time_stamp'][0])>=r['start']-.11 and float(g['time_stamp'][1])<=r['end']+.11]
                if not gg:missing.append({'id':r['id'],'reason':'no_joint_state_relation_teacher','split':v['split'],'task':task});continue
                row={k:r[k] for k in ['id','video','task','segment','step','split','start','end','kind']}
                row.update(state_sentences=sorted({s for g in gg for s in g['attribute_sentence']}),relation_sentences=sorted({s for g in gg for s in g['relation_sentence']}),teacher_records=[{'image':g.get('img_path'),'span':g['time_stamp'],'action_idx':g.get('action_idx')} for g in gg])
                allrows.append(row)
    eligible={}
    for task in cfg['tasks']:
        eligible[task]=sorted({r['step'] for r in allrows if r['task']==task and r['split']=='train' and len({x['video'] for x in allrows if x['task']==task and x['step']==r['step'] and x['split']=='train'})>=cfg['min_fit_videos_per_step']})
    rows=[r for r in allrows if r['step'] in eligible[r['task']]]
    missing.extend({'id':r['id'],'reason':'fewer_than_five_fit_videos','split':r['split'],'task':r['task']} for r in allrows if r not in rows)
    rows=sorted(rows,key=lambda r:(r['task'],r['split'],r['video'],r['segment']))
    assert len({r['id'] for r in rows})==len(rows)
    assert not {r['video'] for r in rows if r['split']=='train'}&{r['video'] for r in rows if r['split']=='val'}
    dump(ROOT/'manifest.json',rows)
    cfg_hash=sha(ROOT/'config.json');protocol_hash=sha(ROOT/'protocol.md')
    sources.update({str(p.relative_to(PROJECT)):sha(p) for p in [CAVE/'manifest.json',CAVE/'remaining-tasks/manifest.json',ROOT/'config.json',ROOT/'protocol.md']})
    byvid=collections.defaultdict(list)
    for i,r in enumerate(rows):byvid[r['video']].append((i,r))
    data={k:[None]*len(rows) for k in ['TailROI','TailGlobal','Background','VideoScene','Identity']};coverage=[]
    # Identity feature vocabulary uses fitting videos only; unseen actor/session stays zero.
    cats=sorted({part for r in rows if r['split']=='train' for part in r['video'].split('_')[1:3]})
    for iv,(vid,rr) in enumerate(byvid.items()):
        task=rr[0][1]['task'];base=CAVE if task in ['quesadilla','pinwheels'] else CAVE/'remaining-tasks'
        metap=base/'features'/f'{vid}.json';meta=read(metap);lookup={r['id']:r for r in meta['rows']}
        ep=base/'embeddings'/f'{vid}.npz';ez=np.load(ep);globs=ez['Global'][:,:1024]
        assert ez['ids'].tolist()==[r['id'] for r in meta['rows']]
        rp=base/'roi'/f'{vid}.json';det=read(rp);masks={(d['uid'],tuple(d['names'])):np.asarray(d['masks'],np.float32) for d in det}
        sources[str(metap.relative_to(PROJECT))]=sha(metap);sources[str(ep.relative_to(PROJECT))]=sha(ep);sources[str(rp.relative_to(PROJECT))]=sha(rp)
        for index,r in rr:
            mr=lookup[r['id']];roi=[];bg=[];glob=[];bad=0;nodet=0
            for uid in mr['frame_uids']:
                fp=base/'features'/f'{uid}.npz';z=np.load(fp);patch=z['patch'].astype('float32');valid=z['valid'];coords=z['coords']
                sources[str(fp.relative_to(PROJECT))]=sha(fp)
                m=masks[(uid,tuple(mr['objects']))].copy();union=m.max(0);nodet+=int(m.sum()==0)
                for slot in range(2):
                    if m[slot].sum()<=0:m[slot]=valid
                w=m/m.sum(-1,keepdims=True);slots=norm(w@patch);center=w@coords
                roi.append(np.r_[slots.ravel(),(center*.25).ravel()])
                gw=valid/valid.sum();glob.append(norm(gw@patch))
                bw=valid*(1-np.clip(union,0,1))
                if bw.sum()<1e-6:bw=valid.copy();bad+=1
                bg.append(norm((bw/bw.sum())@patch))
            scene_mask=np.array([x['step']!=r['step'] for x in meta['rows']]);assert scene_mask.any()
            identity=np.array([float(c in vid.split('_')[1:3]) for c in cats],np.float32)
            data['TailROI'][index]=np.mean(roi,0);data['TailGlobal'][index]=np.mean(glob,0)
            data['Background'][index]=np.mean(bg,0);data['VideoScene'][index]=globs[scene_mask].mean(0);data['Identity'][index]=identity
            coverage.append({'id':r['id'],'frame_uids':mr['frame_uids'],'background_fallback_frames':bad,'no_detection_frames':nodet,'frames':len(roi),'scene_other_segments':int(scene_mask.sum())})
        if (iv+1)%20==0:print('OBSERVATIONS',iv+1,len(byvid),flush=True)
    (ROOT/'features').mkdir(exist_ok=True)
    np.savez_compressed(ROOT/'features/observations.npz',ids=np.array([r['id'] for r in rows]),**{k:np.stack(v) for k,v in data.items()})
    counts={t:{s:{'segments':sum(r['task']==t and r['split']==s for r in rows),'videos':len({r['video'] for r in rows if r['task']==t and r['split']==s})} for s in ['train','val']} for t in cfg['tasks']}
    dump(ROOT/'coverage.json',{'counts':counts,'potential':dict(potential),'eligible_steps':eligible,'excluded':missing,'observations':coverage,'identity_vocabulary':cats,'repeated_teacher_frame_records':repeated,'noise_floor':None,'noise_reason':'No independent repeated teacher annotations or human targets; all-correct rate cannot identify MSE noise floor.'})
    dump(ROOT/'frozen-inputs.json',{'source_sha256':sources,'config_sha256':cfg_hash,'protocol_sha256':protocol_hash,'manifest_sha256':sha(ROOT/'manifest.json'),'observations_sha256':sha(ROOT/'features/observations.npz'),'prepare_sha256':sha(__file__)})
    dump(ROOT/'status.json',{'phase':'observations_prepared','counts':counts,'seconds':time.monotonic()-t0})
    print('PREPARE_COMPLETE',json.dumps(counts),flush=True)

if __name__=='__main__':main()
