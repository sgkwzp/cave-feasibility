"""Finish this authorized run after streaming observations; never launch new methods."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,read,dump

OUT=ROOT/'deployment-mechanism'

def main(pids):
    expected={v['video'] for v in read(OUT/'manifest.json')}
    while True:
        done={p.stem for p in (OUT/'observations').glob('*.json')} & expected
        if len(done)==len(expected):break
        alive=False
        for pid in pids:
            proc=Path(f'/proc/{pid}/cmdline')
            alive=alive or (proc.exists() and b'validate_observe.py' in proc.read_bytes())
        if not alive:
            dump(OUT/'pipeline.json',{'phase':'observation_failed','completed':len(done),'expected':len(expected)})
            raise RuntimeError('Observation process ended before complete coverage; inspect observe.log')
        time.sleep(20)
    for name in ['validate_score.py','validate_analyze.py','validate_report.py']:
        dump(OUT/'pipeline.json',{'phase':name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        path=ROOT/'source'/name
        if not path.exists():raise FileNotFoundError(path)
        p=subprocess.run([sys.executable,'-u',str(path)],stdin=subprocess.DEVNULL)
        if p.returncode:
            dump(OUT/'pipeline.json',{'phase':'failed','source':name,'exit_code':p.returncode});raise SystemExit(p.returncode)
    dump(OUT/'pipeline.json',{'phase':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observe-pid',type=int,nargs='+',required=True);main(p.parse_args().observe_pid)
