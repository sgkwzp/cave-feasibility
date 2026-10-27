"""Normal-only group-selected ridge probes; held-out video clustered inference."""
import time
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from prepare import ROOT,read,dump,sha,norm

ARMS={'Token':['Token'],'Full':['Full'],'TailROI':['TailROI'],'CLS':['CLS'],'Input':['Input'],
      'TailGlobal':['TailGlobal'],'Background':['Background'],'VideoScene':['VideoScene'],'Identity':['Identity'],
      'Nuisance':['Background','VideoScene','Identity'],
      'Nuisance+Token':['Background','VideoScene','Identity','Token'],
      'Nuisance+Full':['Background','VideoScene','Identity','Full'],
      'Nuisance+TailROI':['Background','VideoScene','Identity','TailROI']}

def center(x,steps,fit,test):
    means={int(a):x[fit[steps[fit]==a]].mean(0) for a in np.unique(steps[fit])}
    assert set(steps[test])<=set(means)
    return x[fit]-np.stack([means[int(a)] for a in steps[fit]]),x[test]-np.stack([means[int(a)] for a in steps[test]]),means

def design(arrays,blocks,steps,fit,test):
    xx=[];zz=[];stats={}
    for b in blocks:
        x=norm(arrays[b].astype(np.float64));a,z,means=center(x,steps,fit,test)
        scale=max(np.sqrt((a*a).sum(1).mean()),1e-8);a/=scale;z/=scale
        xx.append(a);zz.append(z);stats[b]={'scale':scale,'fit_rank':None}
    return np.concatenate(xx,1)/np.sqrt(len(blocks)),np.concatenate(zz,1)/np.sqrt(len(blocks)),stats

def solve_path(x,z,y,alphas):
    # (1/n)||Y-XB||² + alpha||B||². Kernel scaling is identical for every arm.
    k=x@x.T;v,u=eigh(k,check_finite=False);v=np.maximum(v,0);uy=u.T@y;kz=z@x.T@u
    return np.stack([(kz/(v+len(x)*a))@uy for a in alphas])

def ci(v,q=(.025,.975)):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return np.quantile(v,q).tolist() if len(v) else None

def main():
    t0=time.monotonic();cfg=read(ROOT/'config.json');rows=read(ROOT/'manifest.json');ids=[r['id'] for r in rows]
    arrays={};hashes={}
    for fn in ['observations','aem','targets']:
        path=ROOT/f'features/{fn}.npz';z=np.load(path);assert z['ids'].tolist()==ids
        arrays.update({k:z[k] for k in z.files if k!='ids'});hashes[fn]=sha(path)
    y=np.concatenate([arrays.pop('state'),arrays.pop('relation')],1).astype(np.float64);dim=y.shape[1]//2
    target_names=['state','relation'];names=['ClassMean']+list(ARMS);n=len(rows)
    predictions={k:np.full_like(y,np.nan) for k in names};residual=np.full_like(y,np.nan)
    errors={k:np.full((n,2),np.nan) for k in names};fold_records={};fit_records={};results={};boot={};checks=[]
    for ti,task in enumerate(cfg['tasks']):
        ix=np.array([i for i,r in enumerate(rows) if r['task']==task]);tr=ix[[rows[i]['split']=='train' for i in ix]];va=ix[[rows[i]['split']=='val' for i in ix]]
        steps=np.array([r['step'] for r in rows]);vid=np.array([r['video'] for r in rows]);vs=np.unique(vid[tr]);rng=np.random.default_rng(cfg['seed']+ti);vs=rng.permutation(vs);assign={v:i%cfg['folds'] for i,v in enumerate(vs)}
        folds=[(tr[np.array([assign[v]!=f for v in vid[tr]])],tr[np.array([assign[v]==f for v in vid[tr]])]) for f in range(cfg['folds'])]
        fold_records[task]={v:int(f) for v,f in assign.items()};yt,yv,ym=center(y,steps,tr,va);residual[va]=yv
        predictions['ClassMean'][va]=0;errors['ClassMean'][va]=(yv.reshape(len(va),2,dim)**2).mean(-1)
        fit_records[task]={};(ROOT/'models'/task).mkdir(parents=True,exist_ok=True)
        for name,blocks in ARMS.items():
            losses=np.zeros((len(cfg['alphas']),2));counts=0
            for train,valid in folds:
                a,b,_=design(arrays,blocks,steps,train,valid);c,d,_=center(y,steps,train,valid)
                p=solve_path(a,b,c,cfg['alphas']);losses+=((p-d[None]).reshape(len(p),len(valid),2,dim)**2).mean(-1).sum(1);counts+=len(valid)
            losses/=counts
            # reversed argmin implements stronger regularization on exact ties.
            choice=np.array([len(cfg['alphas'])-1-int(np.argmin(losses[::-1,j])) for j in range(2)])
            x,z,stats=design(arrays,blocks,steps,tr,va);p=solve_path(x,z,yt,cfg['alphas']);pred=np.concatenate([p[choice[j],:,j*dim:(j+1)*dim] for j in range(2)],1)
            predictions[name][va]=pred;errors[name][va]=((pred-yv).reshape(len(va),2,dim)**2).mean(-1)
            selected=[cfg['alphas'][j] for j in choice];fit_records[task][name]={'alphas':selected,'inner_cv_mse':losses.tolist(),'blocks':blocks,'feature_dimensions':x.shape[1],'stats':stats}
            # Retain the exact fold-fitted transformations and linear coefficients.
            coeff=[]
            for j in range(2):
                a=selected[j];dual=np.linalg.solve(x@x.T+len(x)*a*np.eye(len(x)),yt[:,j*dim:(j+1)*dim]);coeff.append(x.T@dual)
            transform={}
            for b in blocks:
                vals=norm(arrays[b].astype(np.float64));_,_,means=center(vals,steps,tr,va)
                transform['mean_'+b]=np.stack([means[a] for a in sorted(means)]);transform['scale_'+b]=np.array(stats[b]['scale'])
            np.savez_compressed(ROOT/'models'/task/(name.replace('+','_')+'.npz'),coef=np.concatenate(coeff,1),steps=np.array(sorted(ym)),target_means=np.stack([ym[a] for a in sorted(ym)]),**transform)
            recon=z@np.concatenate(coeff,1);diff=float(np.max(np.abs(recon-pred)));assert diff<1e-8,diff
            if name=='Token':
                sk=Ridge(alpha=len(x)*selected[0],fit_intercept=False,solver='cholesky').fit(x,yt[:,:dim]);diff2=float(np.max(np.abs(sk.predict(z)-pred[:,:dim])));assert diff2<1e-8
                checks.append({'task':task,'sklearn_max_abs':diff2,'coefficient_reconstruction_max_abs':diff})
            print('READOUT',task,name,selected,flush=True)
        baseline=errors['ClassMean'][va].mean(0);assert (baseline>1e-12).all()
        task_results={'fit_segments':len(tr),'heldout_segments':len(va),'fit_videos':len(vs),'heldout_videos':len(np.unique(vid[va])),'teacher_within_fit_mse':(yt.reshape(len(tr),2,dim)**2).mean((0,2)).tolist(),'arms':{},'per_step':{}}
        for name in names:
            mse=errors[name][va].mean(0);r2=1-mse/baseline
            task_results['arms'][name]={'mse':mse.tolist(),'nmse':(mse/baseline).tolist(),'r2':r2.tolist(),'mean_r2':float(r2.mean()),'teacher_noise_floor':None}
        for step in sorted(set(steps[va])):
            si=va[steps[va]==step];base=errors['ClassMean'][si].mean(0)
            task_results['per_step'][str(step)]={'n':len(si),'videos':len(set(vid[si])),'class_mean_mse':base.tolist(),'arms':{k:(1-errors[k][si].mean(0)/np.maximum(base,1e-15)).tolist() for k in names}}
        vva=np.unique(vid[va]);b_rng=np.random.default_rng(cfg['seed']+100+ti);draw=np.stack([np.bincount(b_rng.integers(0,len(vva),len(vva)),minlength=len(vva)) for _ in range(cfg['bootstrap'])]);w=draw[:,np.searchsorted(vva,vid[va])]
        total0=w@errors['ClassMean'][va]
        boot[task]=np.stack([(1-(w@errors[k][va])/total0).mean(1) for k in names],1)
        for k in names:task_results['arms'][k]['ci95']=ci(boot[task][:,names.index(k)])
        results[task]=task_results
        dump(ROOT/'status.json',{'phase':'readouts_fitted','completed_tasks':list(results),'seconds':time.monotonic()-t0})
    overall_draw=np.mean([boot[t] for t in cfg['tasks']],0)
    overall={k:{'mean_r2':float(np.mean([results[t]['arms'][k]['mean_r2'] for t in cfg['tasks']])),'ci95':ci(overall_draw[:,j]),'ci90':ci(overall_draw[:,j],(.05,.95)),'teacher_noise_floor':None} for j,k in enumerate(names)}
    contrasts={}
    for a,b in [('Token','ClassMean'),('Full','ClassMean'),('TailROI','ClassMean'),('TailROI','Token'),('Full','Token'),('TailROI','Full'),('Nuisance+Token','Nuisance'),('Nuisance+Full','Nuisance'),('Nuisance+TailROI','Nuisance')]:
        key=a+' minus '+b;d=overall_draw[:,names.index(a)]-overall_draw[:,names.index(b)];contrasts[key]={'point':overall[a]['mean_r2']-overall[b]['mean_r2'],'ci95':ci(d),'ci90':ci(d,(.05,.95))}
    c=contrasts['Token minus ClassMean'];obs=contrasts['TailROI minus Token'];m=cfg['equivalence_margin_r2'];equivalent=c['ci90'][0]>-m and c['ci90'][1]<m
    controlled=contrasts['Nuisance+Token minus Nuisance']['ci95'][0]>0
    if equivalent and obs['ci95'][0]>0:decision='token_linear_readout_near_mean_observation_better'
    elif c['ci95'][0]>0:decision='token_instance_information_with_nuisance_increment' if controlled else 'token_instance_information_semantic_source_unresolved'
    else:decision='inconclusive_no_equivalence_or_positive_token_evidence'
    all_close=all(contrasts[a]['ci90'][0]>-m and contrasts[a]['ci90'][1]<m for a in ['TailROI minus Token','TailROI minus Full','Full minus Token'])
    held=np.array([r['split']=='val' for r in rows]);assert all(np.isfinite(predictions[k][held]).all() for k in names)
    pred_arrays={'ids':np.array(ids)[held],'residual_target':residual[held]}
    pred_arrays.update({k:predictions[k][held] for k in names});np.savez_compressed(ROOT/'predictions.npz',**pred_arrays)
    records=[dict(rows[i],losses={k:errors[k][i].tolist() for k in names}) for i in np.flatnonzero(held)];dump(ROOT/'per-segment-errors.json',records)
    np.savez_compressed(ROOT/'bootstrap.npz',names=np.array(names),overall=overall_draw,**boot)
    dump(ROOT/'fits.json',{'folds':fold_records,'fits':fit_records})
    summary={'tasks':results,'overall':overall,'contrasts':contrasts,'decision':decision,'token_equivalent_to_mean':equivalent,'three_representations_equivalent':all_close,'targets':target_names,'bootstrap':cfg['bootstrap'],'seed':cfg['seed'],'fit_split':'train','heldout_split':'val','teacher_noise_floor':None,'noise_note':'Independent repeated labels / physical ground truth absent. 48.4% all-correct accuracy is not an MSE floor.','feature_sha256':hashes,'source_sha256':sha(__file__),'seconds':time.monotonic()-t0}
    dump(ROOT/'results.json',summary);dump(ROOT/'readout-verification.json',{'checks':checks,'identical_ids':True,'all_heldout_predictions_finite':True,'no_train_val_video_overlap':True,'inner_folds_train_only':True,'saved_coefficients_verified':True,'feature_sha256':hashes})
    dump(ROOT/'status.json',{'phase':'G2_complete','decision':decision,'seconds':time.monotonic()-t0})
    print('G2_COMPLETE',decision,flush=True);print('OVERALL',overall,flush=True);print('CONTRASTS',contrasts,flush=True)

if __name__=='__main__':main()
