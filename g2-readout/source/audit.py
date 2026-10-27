"""Check immutable inputs and inspect observable teacher variation, not noise truth."""
import collections
import numpy as np
from prepare import ROOT,PROJECT,read,dump,sha

def main():
    rows=read(ROOT/'manifest.json');frozen=read(ROOT/'frozen-inputs.json');export=read(ROOT/'export-provenance.json')
    for path,h in frozen['source_sha256'].items():assert sha(PROJECT/path)==h,path
    for task,t in export['tasks'].items():assert sha(PROJECT/f'code/AEM/checkpoints/egoper/{task}.pth.tar')==t['checkpoint_sha256']
    z=np.load(ROOT/'features/targets.npz');assert z['ids'].tolist()==[r['id'] for r in rows];out={}
    for task in sorted(set(x['task'] for x in rows)):
        ix=np.array([i for i,x in enumerate(rows) if x['task']==task and x['split']=='train']);steps=np.array([rows[i]['step'] for i in ix]);o={}
        for k in ['state','relation']:
            y=z[k][ix].astype('float64');res=np.empty_like(y)
            assert np.max(abs(np.linalg.norm(y,axis=1)-1))<1e-5
            for a in sorted(set(steps)):res[steps==a]=y[steps==a]-y[steps==a].mean(0)
            between=y-y.mean(0)
            o[k]={'within_step_mse':float((res**2).mean()),'within_fraction_of_task_variation':float((res**2).sum()/(between**2).sum()),'unique_sentence_sets':len({tuple(rows[i][k+'_sentences']) for i in ix})}
        out[task]=o
    dump(ROOT/'teacher-audit.json',out)
    pred=np.load(ROOT/'predictions.npz');records=read(ROOT/'per-segment-errors.json');assert pred['ids'].tolist()==[r['id'] for r in records]
    maxdiff=0.
    for name in pred.files:
        if name in ['ids','residual_target']:continue
        d=(pred[name]-pred['residual_target']).reshape(len(records),2,-1);loss=(d*d).mean(-1)
        expected=np.array([r['losses'][name] for r in records]);diff=float(abs(loss-expected).max());maxdiff=max(diff,maxdiff);assert diff<1e-12
    dump(ROOT/'audit.json',{'input_files_rehashed':len(frozen['source_sha256']),'five_checkpoint_hashes_unchanged':True,'prediction_error_reconstruction_max_abs':maxdiff,'all_targets_unit_norm':True,'same_frame_independent_reannotations_available':False})
    print('AUDIT_COMPLETE',len(frozen['source_sha256']),maxdiff,flush=True)

if __name__=='__main__':main()
