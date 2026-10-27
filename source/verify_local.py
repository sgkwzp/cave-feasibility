"""Local recomputation from transferred segment scores; no GPU or fitted-model rerun."""
import numpy as np
from sklearn.metrics import roc_auc_score
from common import ROOT,OEV,TASKS,read,dump,sha

def main():
    result=read(ROOT/'summary.json');launch=read(ROOT/'build/launch-inputs.json');cfg=read(ROOT/'execution.json')
    assert cfg['seeds']==[20261027]
    for name,digest in launch['sha256'].items():assert sha(ROOT/name)==digest,name
    for seed in cfg['seeds']:
        meta=read(ROOT/'models'/f'{seed}.json');assert sha(ROOT/'models'/f'{seed}.pt')==meta['checkpoint_sha256']
    errors=[];boot=[];n=0
    for task in TASKS:
        rr=read(ROOT/'results'/task/'scores.json');old={r['id']:r for r in read(OEV/'results'/task/'scores.json')}
        assert all(r['raw']['AEM']==old[r['id']]['raw']['AEM'] for r in rr)
        tests=[r for r in rr if r['split']=='test'];normal_val=[r for r in rr if r['split']=='val']
        n+=len(tests)
        for name in tests[0]['scores']:
            aucs=[]
            for step in result['tasks'][task]['eligible_steps']:
                group=[r for r in tests if r['step']==step]
                aucs.append(roc_auc_score([r['kind']>0 for r in group],[r['scores'][name] for r in group]))
            errors.append(abs(np.mean(aucs)-result['tasks'][task]['macro_auc'][name]))
            th=np.quantile([r['scores'][name] for r in normal_val],.9,method='higher')
            for kind,key in [(False,'fpr'),(True,'recall')]:
                group=[r for r in tests if (r['kind']>0)==kind]
                rate=np.mean([r['scores'][name]>=th for r in group])
                errors.append(abs(rate-result['tasks'][task]['working_points'][name][key]))
        boot.append(np.load(ROOT/'results'/task/'bootstrap.npz')['gain'])
    gain=np.mean(boot,axis=0);valid=np.isfinite(gain)
    ci=np.quantile(gain[valid],[.025,.975])
    assert np.allclose(ci,result['overall']['gain']['ci95'],rtol=0,atol=1e-15)
    assert max(errors)<1e-12
    dump(ROOT/'local-verification.json',{'test_segments':n,'max_metric_reconstruction_error':max(errors),
        'paired_bootstrap_ci_reconstructed':ci.tolist(),'valid_replicates':int(valid.sum()),
        'launch_hashes_and_model_verified':True,'single_seed_verified':True,
        'transfer_archive_sha256':sha(ROOT/'build/results-transfer.tar.gz'),'source_sha256':sha(__file__)})
    print('LOCAL_VERIFIED',n,max(errors),ci.tolist(),flush=True)

if __name__=='__main__':main()
