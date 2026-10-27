"""Render the completed confirmation without altering G1 artifacts."""
import hashlib,json
import numpy as np
from common import ROOT,TASKS,read,dump,sha

OUT=ROOT/'follow-up'
def pct(x):return f'{100*x:.2f}'
def ci(m):return f"{100*m['point']:+.2f} [{100*m['ci95'][0]:+.2f}, {100*m['ci95'][1]:+.2f}]"
def label(k):return k.replace('Learned-centroid','学习质心').replace('ROI-centroid','ROI质心').replace('Learned','学习观察器').replace('Global','全局池化').replace('Duration','时长')

def main():
    r=read(OUT/'results.json');old=read(ROOT/'summary.json');v=read(OUT/'roi-verification.json');cfg=read(OUT/'config.json')
    frozen=read(OUT/'frozen-inputs.json');coverage=read(OUT/'coverage.json');secondary=read(OUT/'secondary-results.json')
    for p,h in frozen['original'].items():assert sha(ROOT/p)==h,p
    # Reconstruct both primary intervals from saved, paired draw arrays.
    reconstructed={}
    for key in ['utility','observer']:
        a,b=r['contrasts'][key]['methods'];dd=[]
        for t in TASKS:
            z=[]
            for name in [a,b]:
                p=OUT/'metric-cache'/(t+'-'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.npz')
                z.append(np.load(p)['draw'][:,0,0])
            dd.append(z[0]-z[1])
        d=np.mean(dd,axis=0);valid=np.isfinite(d);q=np.quantile(d[valid],[.025,.975])
        assert np.allclose(q,r['contrasts'][key]['macro']['auc']['ci95'],rtol=0,atol=1e-14)
        reconstructed[key]={'ci95':q.tolist(),'valid_bootstrap':int(valid.sum())}
    lines=['# 观察器继续实验：时长增量与匹配 ROI 对照','',
      '2026-10-27。单个既有训练种子 20261027；本轮没有新训练。已完成缓存效用分析、固定 ROI、位置质心诊断和录像聚类 bootstrap。', '',
      '## 结论','',
      '物体观测通道值得保留，但学习观察器的必要性没有得到支持。固定 ROI 在 Pinwheels 上的宏 AUROC 更高，两任务平均也更高；当前证据将后续候选收缩到简单的 AEM+时长+固定 ROI。以下完整报告两个口径，保留预先冻结的主指标，不以次要指标替换主指标。','',
      f"- 相对 AEM+时长，三通道的步骤宏 AUROC 平均增量为 **{ci(r['contrasts']['utility']['macro']['auc'])} 点**（95% CI）。",
      f"- 任务内片段混池后再按任务平均，AUROC 增量为 **{ci(r['contrasts']['utility']['pooled']['auc'])} 点**，FPR=10% 的召回增量为 **{ci(r['contrasts']['utility']['pooled']['tpr_interp_10'])} 点**。",
      f"- 学习观察器融合相对匹配固定 ROI 融合，步骤宏 AUROC 增量为 **{ci(r['contrasts']['observer']['macro']['auc'])} 点**。",
      f"- 补充比较：AEM+时长+固定 ROI 相对 AEM+时长，步骤宏 AUROC 增量为 **{ci(secondary['overall']['macro']['auc'])} 点**。这是看到固定臂结果后计算的次要对照，不替换预设主检验。",
      f"- 本轮后续投入门槛：**{'通过' if r['gate_passed'] else '未通过；止于本轮两项确认'}**。历史 G1 仍为未通过；不启动 CEI、条件 GMM、额外种子或 CC4D。",'',
      '前两条分别衡量同一步骤内部排序和跨步骤共用阈值的实际效用。后者有收益不能证明前者普遍提升，也不能将区间跨零解读为没有信息。固定 ROI 并非所有工作点都更强：Pinwheels 的 FPR=10% 召回从 AEM+时长的 22.4% 降到 20.0%，学习三通道为 26.4%。检测性能改善与“效果脑补”机制成立是两个不同的证据要求。若后续继续，应冻结最简单的三通道组合，在未用于选择的任务与预测边界下确认；本轮不继续堆叠复杂模块。','',
      '## 固定口径与数据','',
      '同一 GT 边界和 GT 意图步骤，非背景 Normal 对 Modification/Slip。Quesadilla：232 测试段（172 正常/60 错误，35 录像），4 个既定合格步骤；Pinwheels：468 测试段（343 正常/125 错误，45 录像），9 个既定合格步骤。正常训练/验证共 865 段，全部 1565 段、164 录像。', '',
      '主要指标：合格步骤等权宏 AUROC，再对两任务等权。混池指标：每任务全部测试段共用阈值，再对任务等权。两者均不是 AEM 论文的预测分割帧级 AUC，不能与 73.8 直接相减。', '',
      '融合对正常训练参考校准后的连续分数取等权平均，三通道各占 1/3。正常验证集锁定 90% higher 分位阈值。测试错误标签不参与拟合、checkpoint 选择、校准或权重选择。', '',
      '## 段级步骤宏 AUROC（%）','',
      '| 方法 | Quesadilla | Pinwheels | 两任务平均 |','|---|---:|---:|---:|']
    mainnames=['AEM','AEM+Duration','AEM+Global','AEM+ROI','AEM+Learned','AEM+Duration+ROI','AEM+Duration+Learned']
    for name in mainnames:
        lines.append('| '+label(name)+' | '+' | '.join(pct(r['tasks'][t]['methods'][name]['macro']['auc']['point']) for t in TASKS)+' | '+pct(r['overall'][name]['macro']['auc']['point'])+' |')
    lines+=['','独立观测通道（同一 EVA02、同三帧、同密度）：','',
       '| 通道 | Quesadilla | Pinwheels | 两任务平均 |','|---|---:|---:|---:|']
    for name in ['Global','ROI','Learned','ROI-centroid','Learned-centroid']:
        lines.append('| '+label(name)+' | '+' | '.join(pct(r['tasks'][t]['methods'][name]['macro']['auc']['point']) for t in TASKS)+' | '+pct(r['overall'][name]['macro']['auc']['point'])+' |')
    lines+=['','质心通道保留 4 维绝对位置：正常 train 标准化、逐步骤 LedoitWolf 高斯、留一训练录像距离校准；不做单位归一化和 PCA。它是诊断性对照，不能仅凭高低归因于“状态理解”或排除合法空间错误信号。','',
       '## 配对录像 bootstrap：差值与 95% CI（百分点）','',
       '| 比较 | Quesadilla 宏 AUROC | Pinwheels 宏 AUROC | 两任务宏 AUROC |','|---|---:|---:|---:|']
    for key in ['utility','observer','observer_with_duration','over_aem']:
        a,b=r['contrasts'][key]['methods'];lines.append('| '+label(a)+' − '+label(b)+' | '+' | '.join(ci(r['tasks'][t]['contrasts'][key]['macro']['auc']) for t in TASKS)+' | '+ci(r['contrasts'][key]['macro']['auc'])+' |')
    lines.append('| 固定 ROI 三通道 − AEM+时长（次要对照） | '+' | '.join(ci(secondary['tasks'][t]['macro']['auc']) for t in TASKS)+' | '+ci(secondary['overall']['macro']['auc'])+' |')
    lines+=['','每任务按录像配对重采样 2000 次；缺少某既定步骤正/负类的抽样无效，宏指标联合有效 1999 次。区间条件于固定训练参考和单个已训模型，不包含训练种子或重新拟合的不确定性。','',
       '## 相同实际 FPR 的召回','',
       '下表在每任务所有测试片段上画 ROC，并取共用阈值的 FPR=5/10/20%。这是离线描述性比较；没有将测试阈值用于部署。插值结果与“不超过预算的确定性阈值”结果本次相同，原始 JSON 同时保存两者。','',
       '| 任务 | 方法 | TPR@5% | TPR@10% | TPR@20% | 0–10% pAUC |','|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for name in ['AEM','AEM+Duration','AEM+Learned','AEM+Duration+Learned','AEM+Duration+ROI']:
            m=r['tasks'][task]['methods'][name]['pooled'];lines.append('| '+task+' | '+label(name)+' | '+' | '.join(pct(m[k]['point']) for k in ['tpr_interp_5','tpr_interp_10','tpr_interp_20','pauc_0_10'])+' |')
    lines+=['','pAUC 定义为 FPR [0,0.1] 下 ROC 面积除以 0.1，表中乘 100；随机基线为 5%，不是 sklearn 标准化 pAUC。','',
       '三通道相对 AEM+时长的两任务平均差值：','',
       '| 口径 | AUROC | AP | 0–10% pAUC | TPR@10% |','|---|---:|---:|---:|---:|']
    for scope,title in [('macro','步骤宏平均'),('pooled','任务内混池')]:
        lines.append('| '+title+' | '+' | '.join(ci(r['contrasts']['utility'][scope][k]) for k in ['auc','ap','pauc_0_10','tpr_interp_10'])+' |')
    lines+=['','## 正常验证集固定阈值的部署诊断','',
       '| 任务 | 方法 | 召回 % | 实际 FPR % |','|---|---|---:|---:|']
    for task in TASKS:
        for name in ['AEM','AEM+Duration','AEM+Learned','AEM+Duration+Learned','AEM+Duration+ROI']:
            m=r['tasks'][task]['methods'][name]['deployment'];lines.append('| '+task+' | '+label(name)+' | '+pct(m['recall']['point'])+' | '+pct(m['fpr']['point'])+' |')
    lines+=['','三通道相对 AEM+时长，使用各自的正常验证阈值：','',
       '| 任务 | 新检出（片段/录像） | 丢失检出 | 新误报 | 消除误报 |','|---|---:|---:|---:|---:|']
    for task in TASKS:
        e=read(OUT/'analysis'/task/'events.json')['utility']
        lines.append('| '+task+' | '+' | '.join(str(e[k]['segments'])+'/'+str(e[k]['videos']) for k in ['new_TP','lost_TP','new_FP','resolved_FP'])+' |')
    lines+=['','逐片段 ID 已保存于 `analysis/<task>/events.json`。正常验证录像各仅 3 条，且同一正常验证集曾参与选 checkpoint；此处是经验阈值，不宣称严格共形误报控制。','',
       '## 逐步骤：效用与固定 ROI 对照','',
       '| 任务/步骤 | AEM+时长 | 加学习观察器 | 差值 | AEM+固定 ROI | AEM+学习观察器 |','|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        mm=r['tasks'][task]['methods']
        for step in r['tasks'][task]['eligible_steps']:
            get=lambda name:mm[name]['by_step'][str(step)]['auc']
            vals=[get(n) for n in ['AEM+Duration','AEM+Duration+Learned']]
            lines.append('| '+task+'/'+str(step)+' | '+pct(vals[0])+' | '+pct(vals[1])+' | '+f'{100*(vals[1]-vals[0]):+.2f}'+' | '+pct(get('AEM+ROI'))+' | '+pct(get('AEM+Learned'))+' |')
    lines+=['','步骤 1 两任务均为 Place tortilla on cutting board。逐步差异是开发诊断，不根据测试得分挑步骤、改权重或重新定义主子集。','',
       '## ROI 实现与验证','',
       '固定 Grounding DINO tiny，480/854 输入尺寸，box/text 阈值 0.25/0.2；物体名来自正常训练多数 query。无合格框的槽退回有效图像 patch 均匀池化。与学习观察器共用三帧、冻结 EVA02 patch、2052 维双槽/坐标表示、PCA32、逐步骤收缩高斯和正常参考校准。', '',
       '| 任务 | train 槽覆盖 % | val 槽覆盖 % | test 槽覆盖 % |','|---|---:|---:|---:|']
    for task in TASKS:
        vals=[]
        for split in ['train','val','test']:
            q=[c for c in coverage if c['task']==task and c['split']==split];vals.append(sum(c['available_slots'] for c in q)/(6*len(q)))
        lines.append('| '+task+' | '+' | '.join(pct(x) for x in vals)+' |')
    lines+=['',f"完整 ROI 检测、池化、评分在 RTX 4090 服务器约 {v['seconds']/60:.2f} 分钟。ROI 对照需要测试期检测器；学习观察器不用该检测器，但仍需要本轮 EVA02 视觉编码。该耗时不含上轮已缓存的 EVA02 编码，不能作为完整方法端到端耗时。",'',
       f"四张固定选择的正常训练帧经视觉检查，仅用于工程几何 QA，没有新增语义标注。164 条录像各一段独立 NumPy 重建池化，最大误差 {max(c['max_abs_error'] for c in v['pooling_checks']):.3g}；所有 1565 段 ID/帧引用对齐。",'',
       f"实际分数的 sklearn 独立指标核验最大误差 {r['verification']['max_sklearn_metric_error']:.3g}。两个主比较的区间由保存抽样数组重建通过；20 个原输入/源码/结果文件哈希保持一致；传输包 SHA256 核验通过。",'',
       '首次服务器启动因未同步本地 prepare.py 而在输入哈希检查阶段退出；补齐原文件后成功运行。未进入模型运算，无实验配置或测试驱动参数修改。','',
       '## 后续投入判据','',
       '| 事先冻结条件 | 结果 |','|---|---|']
    for key,value in r['gate'].items():lines.append('| '+key+' | '+('通过' if value else '未通过')+' |')
    lines+=['','本轮可以支持的主张是特定三通道融合在已探索任务上的效用证据，以及固定 ROI 与学习观察器的直接比较。当前不据此宣布完整 CAVE 有效，也不将门槛未过解释为观察信息不存在。条件转移、CEI、无人工标注的真实结果机制归因、预测边界泛化和外部任务均未在本轮验证。','',
       '## 文件','',
       '- `config.json`、`protocol.md`、`frozen-inputs.json`：出新分数前冻结的规则与输入哈希。',
       '- `results.json`：全体方法/步骤、两种口径、指标与区间；`utility-results.json` 为 ROI 完成前的缓存分析。',
       '- `secondary-results.json`：完成固定臂后计算的 ROI 三通道效用差值，未用于改变预设门槛。',
       '- `analysis/<task>/scores.json`、`events.json`：逐片段连续分数与检出/误报变化。',
       '- `roi-results/`、`embeddings/`、`coverage.json`：固定 ROI 分数、向量与覆盖统计。',
       '- `metric-cache/`：可重建配对区间的抽样指标数组。',
       '- `normal-roi-qa.jpg`、`roi-verification.json`、`verification.json`：几何 QA 与核验。','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(OUT/'verification.json',{'original_files_unchanged':True,'model_seed':20261027,'new_training_runs':0,
        'primary_intervals_reconstructed':reconstructed,'max_sklearn_metric_error':r['verification']['max_sklearn_metric_error'],
        'roi_pooling_max_error':max(c['max_abs_error'] for c in v['pooling_checks']),
        'transfer_archive_sha256':sha(OUT/'roi-transfer.tar.gz'),'report_sha256':sha(OUT/'report.md'),
        'report_source_sha256':sha(__file__)})
    dump(OUT/'status.json',{'phase':'completed_follow_up','gate_passed':r['gate_passed'],
        'next':'mechanism_external_confirmation' if r['gate_passed'] else 'stop_expansion_preserve_development_findings',
        'new_training_runs':0,'CEI_GMM_CC4D_launched':False})
    print('REPORT_WRITTEN',str(OUT/'report.md'),flush=True)

if __name__=='__main__':main()
