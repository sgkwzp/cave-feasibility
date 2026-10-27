"""Run the authorized G1 comparison once, stop at its decision, retain failures."""
import datetime,json,os,subprocess,sys,traceback
from common import ROOT,read,dump,sha

def main():
    assert (ROOT/'feature-index.json').exists(),'Complete locked extraction first'
    lock=ROOT/'build/pipeline.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    dump(ROOT/'build/launch-inputs.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pid':os.getpid(),'sha256':{str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'execution.json',ROOT/'manifest.json',ROOT/'supervision.json',ROOT/'feature-signature.json']+sorted((ROOT/'source').glob('*.py'))}})
    try:
        stages=[('train',[]),('evaluate',['--probe']),('detect',[]),('evaluate',[]),('verify',[])]
        stopped=False
        for stage,args in stages:
            if stopped and stage!='verify':continue
            print('START_STAGE',stage,flush=True)
            log=stage+('-probe' if args else '')
            with (ROOT/'build'/f'{log}.log').open('w',encoding='utf-8') as out:
                result=subprocess.run([sys.executable,'-u',str(ROOT/'source'/f'{stage}.py')]+args,stdout=out,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{stage} returned {result.returncode}; see build/{stage}.log')
            print('COMPLETE_STAGE',stage,flush=True)
            if stage=='evaluate' and args and not read(ROOT/'summary.json')['aem_gate_passed']:
                stopped=True;print('EARLY_STOP: necessary AEM gain gate failed; skip detector and all later modules',flush=True)
        print('PIPELINE_COMPLETE',flush=True)
    except Exception as exc:
        dump(ROOT/'status.json',{'phase':'failed','error':str(exc),'traceback':traceback.format_exc()});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
