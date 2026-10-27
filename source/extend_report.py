"""Report all five tasks while isolating new-task evidence."""
import hashlib
import numpy as np
from common import ROOT,read,dump,sha

EXT=ROOT/'remaining-tasks'
def pct(x):return f'{100*x:.2f}'
def ci(x):return f"{100*x['point']:+.2f} [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]"
def label(s):return s.replace('AEM-raw','原始 AEM').replace('Duration','时长').replace('Learned','冻结学习观察器').replace('Global','全局池化')

def main():
    r=read(EXT/'summary.json');f=read(EXT/'frozen-inputs.json');cfg=read(EXT/'config.json');coverage=read(EXT/'coverage.json')
    v=read(EXT/'score-verification.json');fs=read(EXT/'feature-signature.json')
    index=read(EXT/'feature-index.json');manifest=read(EXT/'manifest.json')
    refs={r['id']:r for video in manifest for r in video['rows']};seen=set();checked_frames=0
    for video in index:
        entries={e['uid']:e for e in video['entries']};fps=video['fps'];count=video['count']
        for row in video['rows']:
            assert row['id'] not in seen;seen.add(row['id']);ref=refs[row['id']]
            assert all(row[k]==ref[k] for k in ['id','task','video','step','split','kind','start','end','objects','old_times'])
            upper=count-1
            if row['next_action_start'] is not None:upper=min(upper,int(np.floor((row['next_action_start']-1e-6)*fps)))
            expected=[min(upper,max(0,int(round(t*fps)))) for t in row['old_times']]
            observed=[entries[u]['frame_index'] for u in row['frame_uids']];assert expected==observed
            checked_frames+=len(observed)
    assert seen==set(refs) and len(seen)==3030
    for p,h in f['original'].items():assert sha(ROOT/p)==h,p
    primary=r['aggregates']['new_three']['contrasts']['primary']['macro']['auc'];gates=r['gate']
    boot=[]
    for t in cfg['tasks']:
        values=[]
        for name in cfg['primary_pair']:
            p=EXT/'metric-cache'/(t+'-'+hashlib.sha256(name.encode()).hexdigest()[:12]+'.npz')
            values.append(np.load(p)['draw'][:,0,0])
        boot.append(values[0]-values[1])
    d=np.mean(boot,axis=0);q=np.quantile(d[np.isfinite(d)],[.025,.975]);assert np.allclose(q,primary['ci95'],rtol=0,atol=1e-14)
    positive=[t for t in cfg['tasks'] if r['tasks'][t]['contrasts']['primary']['macro']['auc']['point']>0]
    lines=['# 物体观测通道：补齐 EgoPER 五任务','',
       '2026-10-27。按用户要求补做 Coffee、Tea、Oatmeal；Quesadilla/Pinwheels 复用此前已核验结果。本轮新增训练次数为 0，沿用唯一种子 20261027 的观察器。','',
       '## 直接结论','',
       f"主比较固定为 **AEM+时长+固定 ROI 对 AEM+时长**。新增三任务的步骤宏 AUROC 平均增量为 **{ci(primary)} 点**（录像聚类 95% CI），其中 {len(positive)}/3 个任务点估计为正。",'',
       f"五任务平均增量为 **{ci(r['aggregates']['all_five']['contrasts']['primary']['macro']['auc'])} 点**；旧两任务平均为 **{ci(r['aggregates']['prior_two']['contrasts']['primary']['macro']['auc'])} 点**。判断泛化时以新增三任务为主，不用五任务平均掩盖退化。",'',
       ('本轮冻结的进一步扩展条件通过。可继续验证预测边界和实际结果证据依赖；当前仍只是冻结物体观测与简单融合，不能据此确证完整 CAVE 或“效果脑补”机制。' if r['gate_passed'] else
        '本轮冻结的进一步扩展条件未通过。已完成全部五任务覆盖，停止继续增加复杂模块。保留各任务的正向结果及低误报表现，不将总体未过门槛解释为每个任务都没有效用。'),'',
       '## 为什么先前只有两个任务','',
       '先前用已具备完整观测缓存的 Quesadilla/Pinwheels 完成最小可行性和匹配 ROI 对照，目的在于控制验证成本。该范围不能代表五任务。收到本次补齐要求后，固定方法、校准与比较规则，为其余三任务重新解码和提取观测特征。', '',
       '三任务此前用于 AEM/时长以及其他方向的诊断，故称为“本轮固定方案的跨任务验证”，不称从未接触的独立数据集。旧门槛及失败记录原样保留，本次是用户授权的任务覆盖扩展。','',
       '## 五任务同口径结果：步骤宏 AUROC（%）','',
       '| 任务 | AEM | AEM+时长 | 加全局池化 | 加固定 ROI | 加冻结学习观察器 | ROI 相对 AEM+时长 |',
       '|---|---:|---:|---:|---:|---:|---:|']
    names=['AEM-raw','AEM+Duration','AEM+Duration+Global','AEM+Duration+ROI','AEM+Duration+Learned']
    for task in cfg['all_tasks']:
        x=r['tasks'][task];lines.append('| '+task+' | '+' | '.join(pct(x['methods'][k]['macro']['auc']['point']) for k in names)+' | '+ci(x['contrasts']['primary']['macro']['auc'])+' |')
    for group,title in [('new_three','新增三任务'),('prior_two','旧两任务'),('all_five','全部五任务')]:
        x=r['aggregates'][group];lines.append('| **'+title+'** | '+' | '.join(pct(x['methods'][k]['macro']['auc']['point']) for k in names)+' | '+ci(x['contrasts']['primary']['macro']['auc'])+' |')
    lines+=['','“加”列均是在 AEM+时长上加入对应观测通道，三路权重各 1/3。它们使用同三帧和同 EVA02 编码器。主指标用 GT 边界、GT 意图步骤，Normal 对 Modification/Slip；不是 AEM 论文 73.8 的预测分割帧级指标。','',
       '表中 AEM 单独基准使用未经校准的原始官方分数。所有融合仍使用本轮预设的正常参考平滑校准。核验发现 Oatmeal 步骤 15 的平滑尾概率在少量高分样本上数值饱和为并列，校准 AEM 的宏 AUROC 为 70.61，而原始 AEM 为 70.34。单独补列原始分数恢复真实基准；主比较、融合分数及门槛均未改变。其他方法也保留实际连续分数及其数值并列，不按测试标签破平。','',
       '## 新任务的关键对照','',
       '| 比较 | 新三任务宏 AUROC 差值及 95% CI | 任务内混池 AUROC 差值 |',
       '|---|---:|---:|']
    for key,title in [('primary','固定 ROI 三通道 − AEM+时长'),('over_aem','固定 ROI 三通道 − 原始 AEM'),('roi_vs_global','固定 ROI 三通道 − 全局池化三通道'),('learned_vs_roi','学习观察器三通道 − 固定 ROI 三通道')]:
        x=r['aggregates']['new_three']['contrasts'][key];lines.append('| '+title+' | '+ci(x['macro']['auc'])+' | '+ci(x['pooled']['auc'])+' |')
    lines+=['','学习观察器直接使用仅在 Quesadilla/Pinwheels 正常数据上训练的模型，没有在新任务微调。该结果检验这一个冻结模型的迁移能力，不能据此否定所有目标任务训练的观察器。固定 ROI 的物体名由各任务正常训练记录多数投票确定。','',
       '## 低误报与实际阈值','',
       '下表的 TPR@FPR 在每任务所有测试片段上计算，共用一个任务级阈值。测试 ROC 工作点仅用于离线比较；部署诊断另用正常验证集 90% higher 分位阈值。pAUC 为 FPR [0,0.1] 积分除以 0.1，随机基准为 5%。','',
       '| 任务 | 方法 | TPR@5% | TPR@10% | TPR@20% | 0–10% pAUC | 验证阈值召回 | 实际 FPR |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
    for task in cfg['tasks']:
        for name in ['AEM+Duration','AEM+Duration+ROI','AEM+Duration+Learned']:
            x=r['tasks'][task]['methods'][name];lines.append('| '+task+' | '+label(name)+' | '+' | '.join(pct(x['pooled'][m]['point']) for m in ['tpr_interp_5','tpr_interp_10','tpr_interp_20','pauc_0_10'])+' | '+pct(x['deployment']['recall'])+' | '+pct(x['deployment']['fpr'])+' |')
    lines+=['','固定 ROI 三通道相对 AEM+时长，在新增三任务上的平均差值：','',
       '| 口径 | AUROC | AP | 0–10% pAUC | TPR@10% |','|---|---:|---:|---:|---:|']
    for scope,title in [('macro','同一步骤宏平均'),('pooled','任务内所有片段混池')]:
        x=r['aggregates']['new_three']['contrasts']['primary'][scope];lines.append('| '+title+' | '+' | '.join(ci(x[m]) for m in ['auc','ap','pauc_0_10','tpr_interp_10'])+' |')
    lines+=['','每任务仅 3 条正常验证录像；阈值是经验校准，不能宣称严格共形误报保证。旧两任务的正常验证数据还曾参与观察器选 checkpoint，新三任务验证数据仅用于阈值。','',
       '## 稀疏步骤敏感性','',
       '| 新任务 | 主宏平均步骤数 | ≥8 错误步骤数 | 主比较 AUROC 差值 | ≥8 错误步骤差值 |',
       '|---|---:|---:|---:|---:|']
    for task in cfg['tasks']:
        x=r['tasks'][task];lines.append('| '+task+' | '+str(len(x['eligible_steps']))+' | '+str(len(x['supported_steps']))+' | '+ci(x['contrasts']['primary']['macro']['auc'])+' | '+ci(x['contrasts']['primary']['supported_macro_auc'])+' |')
    lines+=['',f"≥8 错误步骤的新三任务平均差值：**{ci(r['aggregates']['new_three']['contrasts']['primary']['supported_macro_auc'])} 点**。这是预设敏感性分析，不替换主结果。",'',
       f"每任务录像聚类 bootstrap 2000 次。稀疏步骤可能在重采样中缺失正/负类，此时该宏平均样本记为无效而不改变步骤集合；主三任务联合有效 {primary['valid_bootstrap']} 次。该区间条件于保留所有步骤的有效抽样、固定正常参考和固定模型；稀疏步骤越多，解释越受限制。具体每任务有效次数保存在 summary.json。",'',
       '## 数据、观测覆盖与成本','',
       '| 新任务 | train 录像/段 | val 录像/段 | test 录像/段 | 测试错误 | 测试 ROI 槽覆盖 % |',
       '|---|---:|---:|---:|---:|---:|']
    for task in cfg['tasks']:
        count=f['counts'][task];cc=[c for c in coverage if c['task']==task and c['split']=='test'];cover=sum(c['available_slots'] for c in cc)/(6*len(cc))
        lines.append('| '+task+' | '+' | '.join(str(count[s]['videos'])+'/'+str(count[s]['segments']) for s in ['train','val','test'])+' | '+str(r['tasks'][task]['errors'])+' | '+pct(cover)+' |')
    lines+=['',f"新增 225 条录像、3030 段、{fs['frames']} 个唯一取证帧；五任务合计 389 条录像、4595 段，其中测试 2141 段。",'',
       '每段取 85%/90%/95% 时间处的三帧，使用 GT 边界；不声称这些帧都在手撤离后或已呈现最终结果。所有任务正常训练段用于各自 PCA32 与逐步骤收缩高斯，训练距离留一录像计算；正常训练尾概率校准及等权融合不使用测试错误。', '',
       '正常训练帧的固定 QA 样例显示水槽存在漏检，部分杯/碗、茶包、咖啡对象有误定位。框覆盖率只表示产生了匹配文本的框，不等于定位正确率。图像与框的几何转换正常；没有根据这些样例调检测阈值或重选 query。样例保存在 normal-roi-qa.jpg，不构成新增人工标注。', '',
       f"服务器新增三任务提取耗时约 {fs['seconds']/60:.2f} 分钟，ROI 检测、池化及正常密度评分约 {v['seconds']/60:.2f} 分钟。单卡 RTX 4090，冻结观察器严格加载原 checkpoint，新增训练为 0。",'',
       '## 新任务逐步骤','',
       '| 任务/步骤 | 步骤名称 | 正常/错误 | AEM+时长 | 固定 ROI 三通道 | 差值 |',
       '|---|---|---:|---:|---:|---:|']
    for task in cfg['tasks']:
        x=r['tasks'][task]
        for a in x['eligible_steps']:
            b=x['methods']['AEM+Duration']['by_step'][str(a)];o=x['methods']['AEM+Duration+ROI']['by_step'][str(a)]
            lines.append('| '+task+'/'+str(a)+' | '+b['intended'].replace('|','/')+' | '+str(b['normals'])+'/'+str(b['errors'])+' | '+pct(b['auc'])+' | '+pct(o['auc'])+' | '+f"{100*(o['auc']-b['auc']):+.2f}"+' |')
    lines+=['','## 核验与文件','',
       f"旧两任务 AEM/Duration 校准分数重建误差 ≤{max(f['baseline_reconstruction_max_errors'].values()):.3g}；新任务官方 checkpoint 与原导出分数、训练/评测源码哈希匹配。原实验和前轮报告均未改写。",'',
       f"每条新增录像首段独立重建 ROI 池化向量，最大误差 {max(c['max_error'] for c in v['pooling_checks']):.3g}。3030 段与冻结清单逐项一致，{checked_frames} 个帧引用与指定时间/边界舍入规则完全对应。AUROC/AP 对 sklearn 重建最大误差 {r['verification']['max_sklearn_auc_ap_error']:.3g}。主比较置信区间从保存的配对抽样数组重建通过。",'',
       '初次启动因 Windows 路径分隔符导致服务器哈希核对找不到文件，在计算前退出；修复路径序列化后执行。没有修改数据、分数、超参数或纳入规则。','',
       '- `protocol.md`、`config.json`、`frozen-inputs.json`：先于新任务观测分数冻结。',
       '- `summary.json`：五任务与三种汇总口径，指标、区间和冻结判据。',
       '- `results/<task>/scores.json`、`fits.json`：新增任务逐段分数与正常拟合参数。',
       '- `analysis/<task>/results.json`、`events.json`：逐步骤指标、固定验证阈值下的检出/误报变化。',
       '- `metric-cache/`、`verification.json`：重建区间及数值核验。',
       '- 原两任务证据继续保存在 `../follow-up/`；大帧特征与检测缓存保留在服务器。','']
    (EXT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(EXT/'verification.json',{'old_files_unchanged':True,'strict_observer_checkpoint':v['checkpoint_sha256'],
        'new_training_runs':0,'manifest_segments_verified':len(seen),'frame_references_verified':checked_frames,
        'baseline_max_error':max(f['baseline_reconstruction_max_errors'].values()),
        'pooling_max_error':max(c['max_error'] for c in v['pooling_checks']),
        'max_sklearn_metric_error':r['verification']['max_sklearn_auc_ap_error'],'primary_ci_reconstructed':q.tolist(),
        'valid_primary_bootstrap':int(np.isfinite(d).sum()),'report_sha256':sha(EXT/'report.md'),'source_sha256':sha(__file__)})
    dump(EXT/'status.json',{'phase':'five_tasks_completed','gate_passed':r['gate_passed'],'new_training_runs':0,
         'next':'predicted-boundary_and_mechanism_confirmation' if r['gate_passed'] else 'stop_complex_expansion',
         'CEI_GMM_CC4D_launched':False})
    print('FIVE_TASK_REPORT',str(EXT/'report.md'),flush=True)

if __name__=='__main__':main()
