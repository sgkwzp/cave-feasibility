"""Independently reconstruct reported macro AUROCs and normal-threshold working points."""
import collections
import numpy as np
from sklearn.metrics import roc_auc_score
from common import ROOT,OEV,TASKS,read,dump,sha

def main():
    cfg=read(ROOT/'execution.json');summary=read(ROOT/'summary.json');index=read(ROOT/'feature-index.json')
    launch=read(ROOT/'build/launch-inputs.json');unchanged=[]
    for name,digest in launch['sha256'].items():
        assert sha(ROOT/name)==digest,name;unchanged.append(name)
    assert len(index)==164 and sum(len(v['rows']) for v in index)==1565
    errors=[];counts={};supervision=read(ROOT/'supervision.json')
    assert all(t['split'] in ['train','val'] for t in supervision)
    for task in TASKS:
        rows=read(ROOT/'results'/task/'scores.json');old={r['id']:r for r in read(OEV/'results'/task/'scores.json')}
        assert len({r['id'] for r in rows})==len(rows)
        assert all(r['raw']['AEM']==old[r['id']]['raw']['AEM'] and r['scores']['AEM']==old[r['id']]['scores']['AEM'] for r in rows)
        test=[r for r in rows if r['split']=='test'];val=[r for r in rows if r['split']=='val']
        steps=summary['tasks'][task]['eligible_steps'];scores=summary['tasks'][task]['macro_auc']
        recomputed={}
        for name,expected in scores.items():
            if name.endswith('seed_mean'):continue
            vals=[]
            for a in steps:
                rr=[r for r in test if r['step']==a]
                vals.append(roc_auc_score([r['kind']>0 for r in rr],[r['scores'][name] for r in rr]))
            actual=float(np.mean(vals));errors.append(abs(actual-expected));recomputed[name]=actual
            threshold=float(np.quantile([r['scores'][name] for r in val],.9,method='higher'))
            positive=[r for r in test if r['kind']>0];negative=[r for r in test if r['kind']==0]
            recall=np.mean([r['scores'][name]>=threshold for r in positive]);fpr=np.mean([r['scores'][name]>=threshold for r in negative])
            expected_wp=summary['tasks'][task]['working_points'][name]
            assert abs(recall-expected_wp['recall'])<1e-12 and abs(fpr-expected_wp['fpr'])<1e-12
        seedmean=np.mean([recomputed[f'AEM+Learned_{s}'] for s in cfg['seeds']])
        errors.append(abs(seedmean-scores['AEM+Learned_seed_mean']))
        counts[task]={'segments':len(rows),'test':len(test),'normal_val':len(val),'error_test':sum(r['kind']>0 for r in test)}
    assert max(errors)<1e-12,max(errors)
    for v in index:
        old=np.load(OEV/'features'/(v['video']+'.npz'))
        available={e['uid']:e['frame_index'] for e in v['entries'] if e['type']=='tail'}
        assert [[available[u] for u in r['frame_uids']] for r in v['rows']]==old['frame_indices'][:,:3].tolist()
    for seed in cfg['seeds']:
        meta=read(ROOT/'models'/f'{seed}.json')
        assert meta['checkpoint_sha256']==sha(ROOT/'models'/f'{seed}.pt')
        assert meta['normal_train_frames']==1051 and meta['normal_val_frames']==84
    verification={'counts':counts,'max_macro_auc_reconstruction_error':max(errors),'AEM_scores_unchanged':True,
        'all_tail_frame_indices_match_OEV':True,'no_test_teacher_supervision':True,'checkpoint_hashes_verified':True,
        'unchanged_launch_inputs':unchanged,'source_sha256':sha(__file__)}
    dump(ROOT/'verification.json',verification)
    lines=['# CAVE G1：可训练观察器验证结果','',
        '按用户要求使用一个种子、共享正常训练观察器；固定 GT 边界/意图和 OEV 三帧尾窗。数值是同步骤宏段级 AUROC，不是官方帧级 AUC。','',
        '| 方法 | Quesadilla | Pinwheels | 双任务等权 |','|---|---:|---:|---:|']
    for name in ['AEM','AEM+Duration','AEM+O1','AEM+O3','Global','ROI','AEM+Global','AEM+ROI','AEM+Learned_seed_mean']:
        if name not in summary['tasks'][TASKS[0]]['macro_auc']:continue
        values=[summary['tasks'][t]['macro_auc'][name]*100 for t in TASKS]
        lines.append(f'| {name} | {values[0]:.2f} | {values[1]:.2f} | {np.mean(values):.2f} |')
    lines+=['','本轮仅一个种子，表中的 seed_mean 即该种子结果。历史 O1/O3 使用 DINOv2，仅作保留参考；新 Global/Learned（以及若执行的 ROI）使用完全相同的 EVA02 patch。','']
    for key,label in [('gain','相对 AEM'),('observer_gain','相对匹配 EVA 固定 ROI')]:
        if key not in summary['overall']:continue
        d=summary['overall'][key]
        lines.append(f"{label}：{d['point']*100:+.2f} 点，95% 配对录像区间 [{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}]。")
    lines+=['','## 正常校准工作点','', '| 任务 | AEM 召回 / FPR | 学习观察器融合召回 / FPR |','|---|---:|---:|']
    for t in TASKS:
        w=summary['tasks'][t]['working_points'];a=w['AEM'];b=w['AEM+Learned_seed_mean']
        lines.append(f"| {t} | {a['recall']*100:.2f} / {a['fpr']*100:.2f} | {b['recall']*100:.2f} / {b['fpr']*100:.2f} |")
    lines+=['','## 决策','',f"预设 G1 性能门槛：{'通过，仍需结合工作点与机制证据决定下一阶段' if summary['gate_passed'] else '未通过，停止本版扩展，不启动 CEI、条件 GMM 或 CC4D'}。",'',
        '逐项门槛：'+str(summary['gate']),'',
        '## 核验与限制','',
        f"- 164 条录像、1565 个片段，700 个测试片段；AEM 原始分数完全一致，宏 AUROC 独立重建最大差 {max(errors):.3g}。",
        '- 两任务均为此前反复探索的开发任务。区间条件于当前正常参考拟合和单个训练种子，不是独立确认，也不包含种子不确定性。',
        '- 1051 张正常训练老师图、84 张正常验证老师图；原始老师图直接来自官方 RGB 档案。Pinwheels 的合并步骤可包含多个老师子步骤。',
        '- 观察器由框和状态/关系伪标签监督，模型选择仅看正常验证损失；同一正常验证录像也用于经验报警阈值，故不宣称严格共形误报保证。',
        '- 框/文本伪标签可能不描述结果，固定对象查询也可能指向手或工具。训练成功不等于物理状态理解。',
        '- 本轮验证的是可训练观察器，不是完整 CAVE；没有训练 CEI、条件转移模型或重新训练 AEM。',
        '- 本轮没有新增可确证“效果脑补”的 ISR/IG 结论；相关历史诊断仍保持原状态。','',
        '文件：execution.json、preparation.json、feature-signature.json、models/、results/、summary.json、verification.json。']
    if summary['probe_only']:lines+=['','用户要求无效果立即停止：必要的 AEM 增量门槛未通过后，未运行新增固定 ROI 检测器。不能据本轮声称学习观察器已与匹配 EVA ROI 完成比较。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('VERIFICATION_COMPLETE',verification,flush=True)

if __name__=='__main__':main()
