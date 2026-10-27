"""Strict frozen AEM inference and frozen semantic teacher targets. No optimization."""
import gc,importlib.util,os,sys,time
from pathlib import Path
import numpy as np
import torch
from prepare import ROOT,CAVE,PROJECT,read,dump,sha,norm
sys.path.insert(0,str(PROJECT/'code/AEM'))
sys.path.insert(0,str(PROJECT/'code/AEM/libs/utils'))
os.chdir(PROJECT/'code/AEM')
from libs.core import load_config
from libs.datasets import make_dataset
from libs.modeling import make_meta_arch
from libs.modeling.action_model import PrefixPromptTextEncoder
from libs.datasets.clip_encode import action_tokenize_egoper
from libs.utils import fix_random_seed
import libs.datasets.egoper as egoper
import open_clip

def main():
    t0=time.monotonic();cfg0=read(ROOT/'config.json');rows=read(ROOT/'manifest.json');torch.set_num_threads(4);fix_random_seed(cfg0['seed'],include_cuda=True)
    # Reuse exactly validated native-equivalent graph allocation, never execute its main().
    spec=importlib.util.spec_from_file_location('old_export',CAVE.parent/'aem-segmentation-shift/source/export.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    egoper.generate_node_connected=old.preallocated_graph
    weights=next((Path.home()/'.cache/huggingface/hub/models--timm--eva02_large_patch14_clip_336.merged2b_s6b_b61k/snapshots').glob('*/open_clip_model.safetensors'))
    clip,_,prep=open_clip.create_model_and_transforms('EVA02-L-14-336',pretrained=str(weights),device='cuda')
    clip.eval().requires_grad_(False);tokenizer=open_clip.get_tokenizer('EVA02-L-14-336');cm=(clip,tokenizer,prep)
    textp=ROOT/'features/targets.npz'
    if not textp.exists():
        strings=sorted({s for r in rows for key in ['state_sentences','relation_sentences'] for s in r[key]});vec=[]
        with torch.no_grad():
            for i in range(0,len(strings),32):vec.append(torch.nn.functional.normalize(clip.encode_text(tokenizer(strings[i:i+32]).cuda()).float(),dim=-1).cpu().numpy())
        lookup=dict(zip(strings,np.concatenate(vec)));targets={}
        for name,key in [('state','state_sentences'),('relation','relation_sentences')]:targets[name]=np.stack([norm(np.stack([lookup[s] for s in r[key]]).mean(0)) for r in rows])
        np.savez_compressed(textp,ids=np.array([r['id'] for r in rows]),**targets)
        np.savez_compressed(ROOT/'features/text-embeddings.npz',strings=strings,vectors=np.concatenate(vec))
    shadow=CAVE.parent/'aem-segmentation-shift/cache/data';ann=read(shadow/'annotation.json');provenance={}
    for task in cfg0['tasks']:
        rr=[r for r in rows if r['task']==task];out=ROOT/'features/aem'/task;out.mkdir(parents=True,exist_ok=True)
        if (out/'provenance.json').exists() and all((out/f"{r['video']}.npz").exists() for r in rr):
            provenance[task]=read(out/'provenance.json');continue
        cfg=load_config(str(PROJECT/f'code/AEM/configs/{task}.yaml'));cfg['dataset']['root_dir']=str(shadow);cfg['model']['backbone_type']='convGCNTransformer'
        model=make_meta_arch(cfg['model_name'],**cfg['model']);ck=PROJECT/f'code/AEM/checkpoints/egoper/{task}.pth.tar'
        state=torch.load(ck,map_location='cpu')['state_dict'];state={k.removeprefix('module.'):v for k,v in state.items()};model.load_state_dict(state,strict=True);del state
        model.cuda().eval().requires_grad_(False);tokens=action_tokenize_egoper(cm,task,ann[task],device='cuda')
        with torch.no_grad():text=PrefixPromptTextEncoder.encode(cm,tokens,model.action_model.prompt_prefix)@model.action_model.txt_proj
        taskmeta={'checkpoint_sha256':sha(ck),'strict':True,'config':cfg,'checks':[],'videos':[]}
        for split in ['train','val']:
            needed={r['video'] for r in rr if r['split']==split};remaining=[v for v in needed if not (out/f'{v}.npz').exists()]
            if not remaining:continue
            ds=make_dataset(cfg['dataset_name'],False,cm,'infer_'+split,True,**cfg['dataset'])
            for i in range(len(ds)):
                vid=ds.data_list[i]
                if vid not in remaining:continue
                item=ds[i];vr=[r for r in rr if r['video']==vid];ix=[r['segment'] for r in vr];spans=item['segments'][ix];labels=item['labels'][ix]
                assert labels.tolist()==[r['step'] for r in vr]
                assert np.allclose(spans.numpy(),np.array([[r['start'],r['end']] for r in vr])*10)
                capture={}
                def hook(key):
                    def f(module,inp,output):capture[key]=output.detach()
                    return f
                h=[model.effect_model.feature_enhancer.state_effect_proj.register_forward_hook(hook('state')),model.effect_model.feature_enhancer.relation_effect_proj.register_forward_hook(hook('relation')),model.effect_model.feature_enhancer.effect_transformer.register_forward_hook(hook('transformer'))]
                with torch.no_grad():
                    fpn,masks,*_=model.extract_feature([item]);seqs=model.generate_segment_features(fpn,masks,[spans]);z=[];s=[];rel=[];cls=[]
                    for seq in seqs:
                        enhanced=model.effect_model.inference([seq],[torch.ones(1,dtype=torch.long,device='cuda')],model.device)
                        z.append(enhanced[0].mean(0).cpu().numpy());s.append(capture['state'][0].cpu().numpy());rel.append(capture['relation'][0].cpu().numpy());cls.append(capture['transformer'][0,0].cpu().numpy())
                    full=np.stack(z);vis=torch.from_numpy(full).cuda()@model.action_model.vis_proj
                    score=(1-(torch.nn.functional.cosine_similarity(vis,text[labels.cuda()],dim=-1)+1)/2).cpu().numpy()
                for handle in h:handle.remove()
                oldp=CAVE.parent/f'aem-segmentation-shift/cache/scores/{task}/{split}/{vid}.npz';oldz=np.load(oldp)
                expected=oldz['g_enh'][ix,labels.numpy()];difference=float(np.max(np.abs(score-expected)));assert difference<2e-6,(vid,difference)
                np.savez_compressed(out/f'{vid}.npz',ids=np.array([r['id'] for r in vr]),Token=np.concatenate([np.stack(s),np.stack(rel)],1),Full=full,CLS=np.stack(cls),Input=np.stack([seq.mean(0).cpu().numpy() for seq in seqs]),scores=score)
                taskmeta['checks'].append({'video':vid,'historic_score_max_abs':difference});taskmeta['videos'].append(vid)
                dump(ROOT/'status.json',{'phase':'export_aem','task':task,'split':split,'video':vid,'completed_task_videos':len(list(out.glob('*.npz'))),'task_videos':len({r['video'] for r in rr}),'seconds':time.monotonic()-t0})
                print('EXPORTED',task,split,vid,round(time.monotonic()-t0,1),'score_diff',difference,flush=True)
            del ds;gc.collect()
        provenance[task]=taskmeta;dump(ROOT/f'features/aem/{task}/provenance.json',taskmeta)
        del model;gc.collect();torch.cuda.empty_cache()
    collected={k:[] for k in ['Token','Full','CLS','Input']};opened={}
    for r in rows:
        key=(r['task'],r['video'])
        if key not in opened:opened[key]=np.load(ROOT/f"features/aem/{r['task']}/{r['video']}.npz")
        z=opened[key];j=z['ids'].tolist().index(r['id'])
        for k in collected:collected[k].append(z[k][j])
    # Released task checkpoints have different feature dimensions. Each task is
    # probed independently; trailing zeros preserve every norm and kernel exactly.
    native_dimensions={t:{k:int(collected[k][next(i for i,r in enumerate(rows) if r['task']==t)].shape[0]) for k in collected} for t in cfg0['tasks']}
    stacked={k:np.stack([np.pad(a,(0,max(x.shape[0] for x in values)-a.shape[0])) for a in values]) for k,values in collected.items()}
    np.savez_compressed(ROOT/'features/aem.npz',ids=np.array([r['id'] for r in rows]),**stacked)
    source=[PROJECT/'code/AEM'/p for p in ['libs/modeling/effect_model.py','libs/modeling/meta_archs.py','libs/modeling/action_model.py','libs/datasets/egoper.py']]
    dump(ROOT/'export-provenance.json',{'tasks':provenance,'native_dimensions':native_dimensions,'sources':{str(p.relative_to(PROJECT)):sha(p) for p in source},'clip_weights_sha256':sha(weights),'export_source_sha256':sha(__file__),'target_sha256':sha(textp),'aem_sha256':sha(ROOT/'features/aem.npz'),'no_optimization':True,'torch':torch.__version__,'seconds':time.monotonic()-t0})
    dump(ROOT/'status.json',{'phase':'features_complete','seconds':time.monotonic()-t0});print('FEATURES_COMPLETE',flush=True)

if __name__=='__main__':main()
