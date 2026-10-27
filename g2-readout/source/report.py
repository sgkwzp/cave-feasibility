"""Report G2 fixed-cohort probes without converting pseudolabel readout into truth."""
import numpy as np
from prepare import ROOT,read,dump,sha

def pct(x):return f'{100*x:.2f}'
def interval(c):return f'[{pct(c[0])}, {pct(c[1])}]'
def main():
    r=read(ROOT/'results.json');c=read(ROOT/'coverage.json');e=read(ROOT/'export-provenance.json');v=read(ROOT/'readout-verification.json');rows=read(ROOT/'manifest.json')
    labels={'ClassMean':'步骤类均值','Token':'AEM token','Full':'完整 AEM 段表示','TailROI':'真实尾帧固定 ROI','CLS':'投影前 CLS','Input':'输入 FPN 段均值','TailGlobal':'尾帧全局','Background':'尾帧 ROI 外背景','VideoScene':'同录像其它步骤场景','Identity':'参与者／场次','Nuisance':'三类背景／身份控制','Nuisance+Token':'控制＋token','Nuisance+Full':'控制＋完整表示','Nuisance+TailROI':'控制＋尾帧 ROI'}
    decisions={
        'token_linear_readout_near_mean_observation_better':'在预设线性读出与等效界下，token 接近类均值，尾帧观测更强。这个结果支持可读出信息差异，但尚未证明容量瓶颈、信息完全丢失或真实错误上的因果机制。',
        'token_instance_information_with_nuisance_increment':'token 在留出录像上携带超出步骤均值的实例相关信息，且在本次背景／身份代理控制后仍有增量。“token 只恢复出厂均值”被本次结果否定，应删除该问题表述。',
        'token_instance_information_semantic_source_unresolved':'token 在留出录像上优于步骤均值，因此“token 只恢复出厂均值”不成立；控制后增量未获得稳定证据，实例信息是否属于真实效果语义仍未区分。',
        'inconclusive_no_equivalence_or_positive_token_evidence':'token 和完整表示的留出重建均差于步骤类均值，真实尾帧 ROI 也没有稳定超过类均值。观测相对 token 的优势来自后者更差，尚未建立一个能可靠读出实例伪标签差异的观测阳性对照。因此不能归因为“token 只恢复类均值”或容量瓶颈；停止以本次结果支持效果脑补叙事，不追加网络寻找显著结果。'}
    lines=['# G2：状态／关系实例信息读出','', '日期：2026-10-27；种子：20261027；已完成；没有网络训练或新增人工标注。','',decisions[r['decision']],'',
        '## 主要结果','', '数值为扣除拟合集步骤均值后的留出 R²（%）；0 对应类均值，正值表示残差重建改善，负值表示更差。不是错误检测 AUROC。状态／关系等权，五任务等权。','',
        '| 来源 | 平均 R² % | 95% 录像簇区间 | 真实老师噪声下限 |','|---|---:|---:|---|']
    for name in ['ClassMean','Token','Full','TailROI']:
        a=r['overall'][name];lines.append(f"| {labels[name]} | {pct(a['mean_r2'])} | {interval(a['ci95'])} | 不可识别 |")
    lines+=['','| 任务 | 拟合段 / 录像 | 留出段 / 录像 | token | 完整表示 | 尾帧 ROI |','|---|---:|---:|---:|---:|---:|']
    for task,t in r['tasks'].items():lines.append(f"| {task} | {t['fit_segments']} / {t['fit_videos']} | {t['heldout_segments']} / {t['heldout_videos']} | {pct(t['arms']['Token']['mean_r2'])} | {pct(t['arms']['Full']['mean_r2'])} | {pct(t['arms']['TailROI']['mean_r2'])} |")
    lines+=['','## 配对比较','', '| 对比 | ΔR² 百分点 | 95% CI | 90% CI（等效检查） |','|---|---:|---:|---:|']
    for name,a in r['contrasts'].items():lines.append(f"| {name} | {pct(a['point'])} | {interval(a['ci95'])} | {interval(a['ci90'])} |")
    lines+=['',f"预定等效界 ±2 R² 点；token 与均值等效：**{r['token_equivalent_to_mean']}**；token、完整表示、观测三路两两等效：**{r['three_representations_equivalent']}**。置信区间条件于固定 checkpoint 与拟合录像；不包含重新训练或重新抽取拟合集的不确定性。每任务仅 3 条留出录像，区间支持有限，不能称独立论文确认集。",'',
        '## 背景和身份代理','', '| 来源 | 平均 R² % | 95% CI |','|---|---:|---:|']
    for name in ['CLS','Input','TailGlobal','Background','VideoScene','Identity','Nuisance','Nuisance+Token','Nuisance+Full','Nuisance+TailROI']:
        a=r['overall'][name];lines.append(f"| {labels[name]} | {pct(a['mean_r2'])} | {interval(a['ci95'])} |")
    no=sum(x['no_detection_frames'] for x in c['observations']);fb=sum(x['background_fallback_frames'] for x in c['observations']);nf=sum(x['frames'] for x in c['observations'])
    lines+=['',f'背景来自固定 ROI 并集外 patch；{no}/{nf} 张帧没有任何检测，此时“背景”实际包含全图；{fb}/{nf} 张帧因无剩余 patch 回退全图。因此它是背景代理，不是真实像素语义掩膜。同录像场景代理排除目标步骤的所有段，不使用老师标签；未知参与者／场次编码为零。', '',
        '## 老师目标、噪声与覆盖','',
        '目标是发布的 attribute_sentence / relation_sentence，经冻结 EVA02 文本编码器得到的两个语义向量。四路读出预测完全相同的目标，目标均值只从拟合录像估计。它不是 AEM 内部学习的 GNN/视觉联合目标；这里回答伪标签语义能否被读出，不能把向量距离直接解释成物理状态误差。', '',
        f"共 {len(rows)} 个正常段；老师状态和关系都有效才纳入。覆盖缺失 {len(c['excluded'])} 段，逐段排除原因在 coverage.json。复用既有选帧（段内后部三帧）与固定 ROI；没有保证手已撤离。老师原始效果帧和尾帧可能来自不同时间，反映的是跨观测对同一伪目标的预测能力。",'',
        f"发现 {len(c['repeated_teacher_frame_records'])} 组重复帧路径记录。重复存储不等于独立重标；没有同帧独立重复老师输出或人工真值，因此所有方法的真实重建噪声下限均不可识别。48.4% 逐帧全对率不能推导成 MSE 噪声下限。零向量到伪标签的残差基线也不是噪声下限。",'',
        '### 分目标原始 MSE / NMSE','', '| 任务 | 来源 | 状态 MSE | 状态 NMSE | 关系 MSE | 关系 NMSE |','|---|---|---:|---:|---:|---:|']
    for task,t in r['tasks'].items():
        for name in ['ClassMean','Token','Full','TailROI']:
            a=t['arms'][name];lines.append(f"| {task} | {labels[name]} | {a['mse'][0]:.8f} | {a['nmse'][0]:.4f} | {a['mse'][1]:.8f} | {a['nmse'][1]:.4f} |")
    lines+=['','### 同步骤宏敏感性','', '| 来源 | 每任务先同步骤宏、再五任务平均 R² % |','|---|---:|']
    sensitivity={}
    for name in ['ClassMean','Token','Full','TailROI']:
        val=float(np.mean([np.mean([np.mean(s['arms'][name]) for s in t['per_step'].values()]) for t in r['tasks'].values()]));sensitivity[name]=val;lines.append(f'| {labels[name]} | {pct(val)} |')
    audit=read(ROOT/'teacher-audit.json')
    lines+=['','老师类内方差占任务内总方差（拟合集，仅表示目标有变化，不是信号／噪声分解）：','', '| 任务 | 状态 | 关系 |','|---|---:|---:|']
    for task,a in audit.items():lines.append(f"| {task} | {pct(a['state']['within_fraction_of_task_variation'])}% | {pct(a['relation']['within_fraction_of_task_variation'])}% |")
    lines+=['','各 checkpoint 的原生表示维数：','', '| 任务 | token 拼接 | 完整段表示 | CLS | FPN 输入 |','|---|---:|---:|---:|---:|']
    for task,d in e['native_dimensions'].items():lines.append(f"| {task} | {d['Token']} | {d['Full']} | {d['CLS']} | {d['Input']} |")
    lines+=['','各任务独立拟合，统一缓存的尾部零填充不改变范数或核矩阵。维数与读出难度不同，不能把跨任务差异直接归因于 token 容量。', '', '## 实施与复核','',
        '- 五个公开 checkpoint strict 加载，所有新导出连续分数对照历史缓存，最大允许误差 2e−6。没有改变模型权重、步骤映射、尾帧窗口或 ROI 检测。',
        '- 只拟合诊断线性岭读出；正常 train 内五折按录像选择正则，val 不参与选择、中心化或尺度估计。状态与关系分别选正则，所有表示使用相同候选集合。',
        '- 全部输入逐段 ID 对齐；原始目标、中心化目标、各路预测、系数、预处理和 2000 次配对 bootstrap 都保存。闭式实现对照 sklearn Ridge，另核验存盘系数可重建预测。',
        '- 完整表示指官方动作检测器接收的增强段均值；不是对整段时序特征的任意非线性解码。单次线性读出失败不能证明互信息为零。',
        '- 五任务都曾在历史实验中访问；本次 val 是本读出的留出集，不声称从未用于 AEM 官方 checkpoint 选择或历史研究。', '',
        '## 研究决定','',
        '本轮只解决 G2，不能替代真实错误检测门槛。既有预测边界实验中，AEM+时长+ROI 相对原始 AEM 的平均增量仍是 +0.18 点，95% CI [−1.58,+1.79]；保留这一不显著结果。G1 学习观察器未超过匹配固定 ROI 的结论不变。', '',
        '不启动三组件 CAVE、CEI、混合密度或新数据集扩展。若要继续论文，必须先依据本次明确支持的主张重新限定问题，不能把诊断读出提升写成检测提升。', '',
        '## 证据文件','',
        '- protocol.md、config.json：分岔规则与冻结配置。',
        '- results.json：逐任务、目标、步骤和区间；fits.json：训练内选择与视频折。',
        '- predictions.npz、per-segment-errors.json、bootstrap.npz：可重算连续证据。',
        '- frozen-inputs.json、export-provenance.json、verification.json：来源哈希与校验。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    maxdiff=max(x['historic_score_max_abs'] for t in e['tasks'].values() for x in t['checks'])
    frozen=read(ROOT/'frozen-inputs.json');assert sha(ROOT/'config.json')==frozen['config_sha256'];assert sha(ROOT/'protocol.md')==frozen['protocol_sha256']
    assert sha(ROOT/'manifest.json')==frozen['manifest_sha256']
    for name,h in r['feature_sha256'].items():assert sha(ROOT/f'features/{name}.npz')==h
    dump(ROOT/'verification.json',{'status':'complete','count':len(rows),'heldout_segments':sum(x['split']=='val' for x in rows),'strict_checkpoints':list(e['tasks']),'max_historic_score_error':maxdiff,'readout_checks':v,'config_unchanged':True,'protocol_unchanged':True,'manifest_unchanged':True,'noise_floor_identifiable':False,'step_macro_sensitivity':sensitivity,'report_sha256':sha(ROOT/'report.md'),'results_sha256':sha(ROOT/'results.json')})
    print('REPORT_COMPLETE',r['decision'],maxdiff,flush=True)

if __name__=='__main__':main()
