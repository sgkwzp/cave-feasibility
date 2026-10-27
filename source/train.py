"""Three normal-only shared observers; checkpoint selection uses normal validation supervision."""
import json,random,time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from common import ROOT,read,dump,sha
from observer import Observer,loss_terms

class NormalFrames(Dataset):
    def __init__(self,entries,text):
        self.entries=entries;self.text=text;self.cache=[]
        for e in entries:
            t=e['teacher'];z=np.load(ROOT/'features'/(e['uid']+'.npz'))
            out={k:torch.from_numpy(z[k].copy()) for k in ['patch','valid','coords','teacher_masks']}
            out['query']=torch.from_numpy(np.stack([text['a photo of '+x.lower()] for x in t['names']]))
            for key,field in [('attr','attribute_sentence'),('rel','relation_sentence')]:
                values=[text[x] for x in t[field]]
                target=np.mean(values,axis=0) if values else np.zeros(768,dtype=np.float32)
                if values:target=target/max(np.linalg.norm(target),1e-12)
                out[key+'_target']=torch.from_numpy(target.astype('float32'))
                out[key+'_valid']=torch.tensor(bool(values),dtype=torch.float32)
            self.cache.append(out)
    def __len__(self):return len(self.cache)
    def __getitem__(self,i):return self.cache[i]

def main():
    torch.set_num_threads(4);cfg=read(ROOT/'execution.json');started=time.monotonic()
    z=np.load(ROOT/'text-embeddings.npz');text=dict(zip(z['strings'].tolist(),z['vectors']))
    videos=read(ROOT/'feature-index.json');entries={s:[e for v in videos if v['split']==s for e in v['entries'] if e['type']=='teacher'] for s in ['train','val']}
    assert not set(e['teacher']['video'] for e in entries['train'])&set(e['teacher']['video'] for e in entries['val'])
    datasets={s:NormalFrames(e,text) for s,e in entries.items()}
    directory=ROOT/'models';directory.mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        if (directory/f'{seed}.json').exists():continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
        model=Observer().cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        loaders={s:DataLoader(d,batch_size=32,shuffle=s=='train',num_workers=0) for s,d in datasets.items()}
        history=[];best=float('inf');best_epoch=None
        for epoch in range(cfg['epochs']):
            stats={}
            for split,loader in loaders.items():
                model.train(split=='train');total=0.;n=0;boxmass=[];overlap=[]
                with torch.set_grad_enabled(split=='train'):
                    for raw in loader:
                        b={k:v.cuda().float() for k,v in raw.items()}
                        output=model(b['patch'],b['query'],b['valid'],b['coords']);loss,terms=loss_terms(output,b)
                        assert torch.isfinite(loss)
                        if split=='train':
                            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                        total+=float(loss.detach())*len(b['patch']);n+=len(b['patch'])
                        alpha=output[1].detach();available=b['teacher_masks'].sum(-1)>0
                        mass=(alpha*(b['teacher_masks']>0)).sum(-1)
                        boxmass.extend(mass[available].cpu().tolist())
                        overlap.extend(torch.minimum(alpha[:,0],alpha[:,1]).sum(-1).cpu().tolist())
                stats[split]={'loss':total/n,'target_box_mass':float(np.mean(boxmass)),'slot_overlap':float(np.mean(overlap)),'frames':n}
            if stats['val']['loss']<best:
                best=stats['val']['loss'];best_epoch=epoch+1
                torch.save({'state_dict':model.state_dict(),'seed':seed,'epoch':epoch+1,'stats':stats},directory/f'{seed}.pt')
            history.append({'epoch':epoch+1,**stats})
            if (epoch+1)%10==0:
                status={'phase':'training','seed':seed,'epoch':epoch+1,'best_epoch':best_epoch,'normal_val_loss':stats['val']['loss'],'seconds':time.monotonic()-started}
                dump(ROOT/'status.json',status);print(json.dumps(status),flush=True)
        dump(directory/f'{seed}.json',{'seed':seed,'best_epoch':best_epoch,'history':history,
            'normal_train_frames':len(datasets['train']),'normal_val_frames':len(datasets['val']),
            'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(directory/f'{seed}.pt'),
            'source_sha256':{'train.py':sha(__file__),'observer.py':sha(ROOT/'source/observer.py')},
            'selection':'minimum normal-validation supervision loss; no test features or error labels read'})
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
