"""Describe ROI utility from completed fixed arms; not a replacement primary test."""
import hashlib
import numpy as np
from common import ROOT,TASKS,read,dump,sha
from confirm_analysis import METRICS,metric_dict

OUT=ROOT/'follow-up'
def main():
    r=read(OUT/'results.json');a,b='AEM+Duration+ROI','AEM+Duration';tasks={};points=[];draws=[]
    for t in TASKS:
        p=[];d=[]
        for name in [a,b]:
            p.append(np.array([[r['tasks'][t]['methods'][name][scope][k]['point'] for k in METRICS] for scope in ['macro','pooled']]))
            f=OUT/'metric-cache'/(t+'-'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.npz')
            d.append(np.load(f)['draw'])
        point=p[0]-p[1];draw=d[0]-d[1];points.append(point);draws.append(draw)
        tasks[t]={scope:metric_dict(point[j],draw[:,j]) for j,scope in enumerate(['macro','pooled'])}
    point=np.mean(points,axis=0);draw=np.mean(draws,axis=0)
    result={'methods':[a,b],'role':'secondary descriptive contrast, computed after inspecting fixed-arm results; does not replace either primary or change gate',
            'tasks':tasks,'overall':{scope:metric_dict(point[j],draw[:,j]) for j,scope in enumerate(['macro','pooled'])},
            'parent_results_sha256':sha(OUT/'results.json'),'source_sha256':sha(__file__)}
    dump(OUT/'secondary-results.json',result)
    print('ROI_UTILITY',result['overall']['macro']['auc'],flush=True)

if __name__=='__main__':main()
