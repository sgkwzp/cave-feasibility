"""Evidence-linked report and independent verification, without modifying old stages."""
import datetime,json
import numpy as np
from common import ROOT,PROJECT,read,dump,sha

OUT=ROOT/'deployment-mechanism'
def fmt(x):return f'{100*x:.2f}'
def ci(x):return f"{100*x['point']:+.2f} [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]"

def main():
    s=read(OUT/'summary.json');cfg=read(OUT/'config.json');frozen=read(OUT/'frozen-inputs.json')
    from validate_qa import main as render_qa
    render_qa()
    assert len(read(OUT/'normal-window-qa.json'))==15
    for p,h in frozen['files'].items():assert sha(PROJECT/p)==h,p
    for key,name in [('manifest_sha256','manifest.json'),('config_sha256','config.json'),('protocol_sha256','protocol.md')]:assert frozen[key]==sha(OUT/name)
    workers=[read(p) for p in sorted((OUT/'build').glob('worker-*-signature.json'))]
    assert len(workers)==3 and {w['worker'] for w in workers}=={0,1,2}
    assert all(w['source_sha256']==sha(ROOT/'source/validate_observe.py') and w['workers']==3 for w in workers)
    original_signature=read(OUT/'observation-signature.json')
    assert all(all(w[k]==v for k,v in original_signature.items() if k!='source_sha256') for w in workers)
    counts={};coverage={};duplicate={};anchors=[]
    for v in read(OUT/'manifest.json'):
        meta=read(OUT/'observations'/(v['video']+'.json'));z=np.load(OUT/'embeddings'/(v['video']+'.npz'))
        assert z['ids'].tolist()==[r['id'] for r in v['rows']]
        assert z['ROI'].shape==(len(v['rows']),2052) and np.isfinite(z['ROI']).all()
        anchors.append(meta['anchor_max_error'])
        for r in meta['rows']:
            key=f"{v['task']}:{r['arm']}:{v['split']}";counts[key]=counts.get(key,0)+1
            coverage.setdefault(key,[0,0]);coverage[key][0]+=r['available_slots'];coverage[key][1]+=6
            duplicate[key]=duplicate.get(key,0)+(len(set(r['frame_indices']))<3)
            assert len(r['frame_uids'])==3
            assert all(0<=ix<meta['frame_count'] for ix in r['frame_indices'])
            if r['arm']=='Pre':assert all(ix/meta['fps']<=r['start']+.5/meta['fps'] for ix in r['frame_indices'])
            else:assert all(ix/meta['fps']<=r['end']+1e-6 for ix in r['frame_indices'])
    # Rebuild the two primary confidence intervals from the saved paired draws.
    draw={};point={}
    for arm,scope,names in [('P','frame',['AEM+Duration+ROI','AEM+Duration']),('M','macro',['AEM+Duration+Post','AEM+Duration+Pre'])]:
        samples=[];pts=[]
        for t in cfg['tasks']:
            z=np.load(OUT/'metric-cache'/(t+'.npz'))
            samples.append(z[f'{arm}|{names[0]}|{scope}'][:,0]-z[f'{arm}|{names[1]}|{scope}'][:,0])
            pts.append(s['tasks'][t][arm]['contrasts']['primary'][scope]['auc']['point'])
        a=np.mean(samples,axis=0);valid=np.isfinite(a);qq=np.quantile(a[valid],[.025,.975])
        ref=s['aggregates']['all_five'][arm]['contrasts']['primary'][scope]['auc']
        assert np.allclose(qq,ref['ci95'],atol=1e-12) and abs(np.mean(pts)-ref['point'])<1e-12
        draw[arm]={'ci95':qq.tolist(),'valid_bootstrap':int(valid.sum())};point[arm]=ref
    a=s['aggregates']['all_five'];three=s['aggregates']['three_identity_clear']
    lines=['# 五任务预测分割与前态对照结果','',f"日期：{datetime.date.today()}。五任务389条录像；单种子20261027，新增网络训练0次。",
           '',f"预设判定：预测部署P **{'PASS' if s['gates']['P'] else 'NOT_MET'}**；后态特异性M **{'PASS' if s['gates']['M'] else 'NOT_MET'}**。",
           '',f"预测分割下，AEM+Duration+ROI相对AEM+Duration的全帧精确AUROC增量为 **{ci(point['P'])}** 个百分点。",
           f"但相对未重新校准的原始AEM，三路融合仅 **{ci(a['P']['contrasts']['vs_raw_aem']['frame']['auc'])}**；未证明端到端胜过原版。这是本轮下一步决策的关键限制。",
           f"固定GT执行分数时，后态融合相对前态融合的步骤宏AUROC差为 **{ci(point['M'])}** 个百分点。",
           '', '## 1. 预测边界与预测类别', '',
           '表中为全帧精确AUROC（%），包含背景、Correction及官方所有错误种类。正常训练录像的预测片段用于建库，测试GT不参与查询、边界、参考拟合或融合。', '',
           '|任务|原AEM|AEM+Duration|AEM+ROI|AEM+Duration+ROI|后三路−AEM+Duration，95%CI|',
           '|---|---:|---:|---:|---:|---|']
    for t in cfg['tasks']:
        p=s['tasks'][t]['P'];mm=p['methods']
        lines.append('|'+t+'|'+'|'.join(fmt(mm[n]['frame']['auc']['point']) for n in ['AEM-raw','AEM+Duration','AEM+ROI','AEM+Duration+ROI'])+'|'+ci(p['contrasts']['primary']['frame']['auc'])+'|')
    lines += ['|五任务等权|'+'|'.join(fmt(a['P']['methods'][n]['frame']['auc']['point']) for n in ['AEM-raw','AEM+Duration','AEM+ROI','AEM+Duration+ROI'])+'|'+ci(point['P'])+'|',
              '',f"Coffee/Quesadilla/Tea三任务增量：{ci(three['P']['contrasts']['primary']['frame']['auc'])}。Pinwheels/Oatmeal保留已记录的标签对应问题，没有用测试标签重映射。",
              f"同三任务相对原始AEM：{ci(three['P']['contrasts']['vs_raw_aem']['frame']['auc'])}。正常参考逐类校准后的AEM五任务均值为{fmt(a['P']['methods']['AEM']['frame']['auc']['point'])}，低于原始AEM的{fmt(a['P']['methods']['AEM-raw']['frame']['auc']['point'])}；加入时长后为{fmt(a['P']['methods']['AEM+Duration']['frame']['auc']['point'])}。因此观测增量不能全部等同为超过原版的净收益。Pinwheels的大幅净提升还混有类别校准效应和历史标签问题。",
              '', '## 2. 前态与后态的匹配比较','',
              '相同GT边界、GT步骤、AEM/时长分数、冻结模型、物体查询和三帧预算。后态为段内85%、90%、95%；前态为开始前0.6/0.4/0.2秒，边界处截断。每臂分别从正常train拟合PCA32和逐步骤收缩高斯。未把前态放进后态参考分布。', '',
              '|任务|AEM+Duration|+前态ROI|+后态ROI|后态−前态，95%CI|前态独立|后态独立|','|---|---:|---:|---:|---|---:|---:|']
    for t in cfg['tasks']:
        m=s['tasks'][t]['M'];mm=m['methods']
        lines.append('|'+t+'|'+'|'.join(fmt(mm[n]['macro']['auc']['point']) for n in ['AEM+Duration','AEM+Duration+Pre','AEM+Duration+Post'])+'|'+ci(m['contrasts']['primary']['macro']['auc'])+'|'+'|'.join(fmt(mm[n]['macro']['auc']['point']) for n in ['Pre','Post'])+'|')
    lines += ['|五任务等权|'+'|'.join(fmt(a['M']['methods'][n]['macro']['auc']['point']) for n in ['AEM+Duration','AEM+Duration+Pre','AEM+Duration+Post'])+'|'+ci(point['M'])+'|'+'|'.join(fmt(a['M']['methods'][n]['macro']['auc']['point']) for n in ['Pre','Post'])+'|',
              '',f"前态融合相对AEM+Duration：{ci(a['M']['contrasts']['pre_increment']['macro']['auc'])}。后态融合相对AEM+Duration：{ci(a['M']['contrasts']['post_increment']['macro']['auc'])}。",
              f"每步骤至少8个错误的后态−前态敏感性：{ci(a['M']['contrasts']['primary']['supported'])}。",
              '', '这不是效果因果干预：动作前可能仍有上一步交互，段末可能仍被手遮挡。比较回答的是这两个实际窗口的信息价值，不能直接证明token脑补或期望−观测残差的必要性。',
              '', '## 3. 对齐原GT片段集合后的预测输入结果','',
              '将完整预测分数图在原2141个GT合格片段内取均值，固定原38个步骤进行宏平均；不丢弃错分或低覆盖片段。与全帧表的权重和错误范围不同。', '',
              '|任务|预测AEM+Duration|预测+ROI|增量95%CI|','|---|---:|---:|---|']
    for t in cfg['tasks']:
        p=s['tasks'][t]['P'];mm=p['methods'];lines.append(f"|{t}|{fmt(mm['AEM+Duration']['macro']['auc']['point'])}|{fmt(mm['AEM+Duration+ROI']['macro']['auc']['point'])}|{ci(p['contrasts']['primary']['macro']['auc'])}|")
    lines += ['', '## 4. 官方评测口径及阈值限制','',
              '下面直接调用官方metric.py；融合异常分数s转为exp(−s)正常相似度，背景按官方强制正常。40阈值AUC近似和阈值平均EDA依赖分数尺度，不能代替精确AUROC。exp映射可数值饱和；精确表直接使用未截断s。', '',
              '|任务|原AEM官方AUC/EDA|AEM+Duration官方AUC/EDA|三路官方AUC/EDA|正常val阈值测试FPR：二路→三路|','|---|---|---|---|---|']
    for t in cfg['tasks']:
        p=s['tasks'][t]['P'];o=p['official'];vals=[f"{fmt(o[n]['auc'])}/{fmt(o[n]['eda'])}" for n in ['AEM-raw','AEM+Duration','AEM+Duration+ROI']]
        fpr=[fmt(p['methods'][n]['fixed_threshold_frame']['fpr']) for n in ['AEM+Duration','AEM+Duration+ROI']]
        lines.append('|'+t+'|'+'|'.join(vals)+'|'+'→'.join(fpr)+'|')
    lines += ['', '阈值为各任务正常validation预测非BG片段分数90%分位数（higher）；误报率在完整测试时间轴上统计。每任务仅3条正常validation，不主张严格共形保证。精确时间轴保留单帧预测游程；官方映射会跳过这些游程，原AEM官方网格结果与历史原生缓存逐任务重建一致。',
              '', '## 5. 覆盖与核验','', '|任务|前态test可用框槽%|预测test可用框槽%|前态重复帧片段|预测重复帧片段|错误帧→预测BG%|','|---|---:|---:|---:|---:|---:|']
    for t in cfg['tasks']:
        c=s['tasks'][t]['P']['coverage'];den=sum(x['error_frames'] for x in c);bg=sum(x['error_to_bg'] for x in c)/den
        keys=[f'{t}:{arm}:test' for arm in ['Pre','Pred']];cov=[fmt(coverage[k][0]/coverage[k][1]) for k in keys]
        lines.append('|'+t+'|'+'|'.join(cov)+'|'+'|'.join(str(duplicate[k]) for k in keys)+f'|{fmt(bg)}|')
    lines += ['',f"逐录像旧ROI锚点重建最大误差：{max(anchors):.3g}。389条录像共{sum(counts.values())}个新观测臂片段，原始片段ID、帧索引、预测段分数和框均保留。可用框率不等于定位正确率。",
              '正常训练帧抽查中，Coffee的量杯查询定位到了旁边的蓝杯；Oatmeal的碗查询定位到了香蕉碗。框几何正确但存在语义误配，未据此修改查询或检测阈值。前后窗口差异还包含遮挡和定位质量差异，不能仅解释为结果语义的因果贡献。抽查图见[正常窗口对照](normal-window-qa.jpg)。',
              f"主P区间有效bootstrap {draw['P']['valid_bootstrap']}/2000；主M区间 {draw['M']['valid_bootstrap']}/2000。固定步骤缺一类即该次宏平均无效，区间条件于有效重采样。区间不覆盖正常参考拟合、校准集或模型种子的不确定性。",
              '五任务测试集此前已用于多轮诊断和方向选择。本轮冻结新增比较避免了这轮按测试结果调参，但不是全新的独立确认集；后续方法主张仍需独立数据验证。',
              '', '## 6. 决策','']
    lines.append('预测输入下，相对固定AEM+Duration对照的观测增量通过预设门槛。原始AEM比较仍不显著，当前等权校准融合方案未证明端到端优势；暂停把它扩展为完整方法或第二数据集大实验。' if s['gates']['P'] else '预测输入下未通过预设部署门槛，停止扩展该固定ROI部署方案；本轮不调权重、阈值或继续加入模块。')
    lines.append('后态相对前态的优势通过预设门槛，支持继续研究末段观测的信息增量；完整CAVE及效果脑补机制仍需另证。' if s['gates']['M'] else '后态特异性未通过预设门槛，已有收益不能归因于效果核验；不据此启动转移模型、CEI或完整CAVE。')
    lines += ['', '## 复查入口','', '[预设协议](protocol.md)、[输入哈希](frozen-inputs.json)、[完整统计](summary.json)、[核验](verification.json)。逐片段分数在scores/，逐录像观测在observations/，bootstrap draws在metric-cache/；旧阶段文件保持原样。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification={'old_inputs_unchanged':True,'observed_videos':len(anchors),'counts':counts,'max_anchor_error':max(anchors),
                  'primary_reconstructed':draw,'gates':s['gates'],'report_sha256':sha(OUT/'report.md'),
                  'max_metric_error':max(max(r['M']['max_sklearn_error'],r['P']['max_sklearn_error'],r['P']['max_frame_sklearn_error']) for r in s['tasks'].values()),
                  'source_sha256':sha(__file__),'observation_worker_signatures':workers,'qa_sha256':sha(OUT/'normal-window-qa.jpg')}
    dump(OUT/'verification.json',verification);dump(OUT/'status.json',{'phase':'complete','gates':s['gates'],'training_runs':0})
    print('REPORT_COMPLETE',s['gates'],flush=True)

if __name__=='__main__':main()
