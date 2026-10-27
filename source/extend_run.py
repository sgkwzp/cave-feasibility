"""Run only the user-authorized three-task extension."""
import sys,traceback
from common import ROOT,PROJECT,OEV,read,dump,sha
sys.path.insert(0,str(OEV/'cache/deps'))
EXT=ROOT/'remaining-tasks'

def check():
    f=read(EXT/'frozen-inputs.json')
    for p,h in f['original'].items():assert sha(ROOT/p)==h,p
    for p,h in f['source_inputs'].items():assert sha(PROJECT/p)==h,p
    assert sha(EXT/'config.json')==f['config_sha256']
    assert sha(EXT/'manifest.json')==f['manifest_sha256']
    for t,s in f['aem_signatures'].items():
        assert sha(PROJECT/f'code/AEM/checkpoints/egoper/{t}.pth.tar')==s['checkpoint_sha256'],t
        for p,h in s['source_sha256'].items():assert sha(PROJECT/'code/AEM'/p)==h,p

def main():
    check()
    dump(EXT/'launch.json',{'scripts':{p.name:sha(p) for p in (ROOT/'source').glob('extend_*.py')},
         'configuration_sha256':sha(EXT/'config.json'),'normal_only_baseline_match':read(EXT/'frozen-inputs.json')['baseline_reconstruction_max_errors']})
    import extend_extract
    extend_extract.main()
    import torch,gc
    gc.collect();torch.cuda.empty_cache()
    import extend_score
    extend_score.main()
    check();print('THREE_TASKS_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(EXT/'status.json',{'phase':'execution_error','error':repr(exc)});traceback.print_exc();raise
