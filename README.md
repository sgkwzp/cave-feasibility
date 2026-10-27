# CAVE feasibility — test-time tail ROI observation

当前冻结版本：**`v2026.10.27-tail-roi-evidence`**。

本仓库保存 2026-10-27 完成的、在五任务比较中显示增量的**测试时末段固定 ROI 观测**源码。最新有效入口是 `source/validate_*.py`，对应 `deployment-mechanism/`。它不是完整 CAVE 已成功的声明；当前决策是**保留末段观测方向，暂停融合版本扩展**。

## 已验证的增益及边界

五个 EgoPER 任务等权，预测边界 + 预测类别，全帧精确 AUROC（%）：

| 方法 | AUROC |
| --- | ---: |
| 原始 AEM | 72.35 |
| 正常参考校准后的 AEM | 70.88 |
| AEM + Duration | 70.63 |
| AEM + ROI | 72.71 |
| AEM + Duration + ROI | 72.53 |

- 三路相对 **AEM + Duration**：**+1.91 pp，95% 录像簇 bootstrap CI [+1.00, +2.79]**。
- 三路相对**原始 AEM**：**+0.18 pp，CI [−1.58, +1.79]**，未证明显著优势。
- 匹配 GT 边界、步骤、执行分数及三帧预算时，末段 ROI 融合相对前态 ROI 融合：**+4.48 pp，CI [+2.48, +6.61]**；该行是步骤宏段级 AUROC，不能与上表全帧 AUROC 混用。

完整结果、协议和核验见 [报告](deployment-mechanism/report.md)、[冻结协议](deployment-mechanism/protocol.md)、[统计结果](deployment-mechanism/summary.json)、[核验记录](deployment-mechanism/verification.json)。389 条录像，1 个种子 `20261027`，本轮新增网络训练 0 次。

## 使用哪个版本

| 路线 | 源码 | 定位 |
| --- | --- | --- |
| 当前冻结的观测增量验证 | `source/validate_*.py` | **本次上传主版本**：五任务预测输入，以及匹配前态/末段对照 |
| 五任务 GT 扩展 | `source/extend_*.py` | 主版本的前置阶段，结果在 `remaining-tasks/` |
| 两任务固定 ROI 对照 | `source/confirm_*.py` | 历史前置阶段，结果在 `follow-up/` |
| 学习观察器 | `source/train.py`, `observer.py`, `run_pipeline.py` | 历史探索；没有确立优于匹配固定 ROI 的方法优势；**不是默认启动入口** |
| G2 语义读出 | `g2-readout/source/` | 已完成的机制诊断；不支持“效果脑补”立题，不属于增益版本 |

`observer.py` 同时提供固定 ROI 使用的 `pool_vector`，因此保留该模块不意味着主版本训练了观察器。`follow-up-plan.md` 中的新匹配分析尚未执行，不是已实现的新方法。

## 固定算法

复用官方 AEM，冻结 EVA02-L-14-336 与 Grounding DINO tiny。每步物体对由正常训练集固定；取段内 85%、90%、95% 三帧。两路归一化 ROI 特征及位置形成 2052 维表示，经正常训练 PCA32、逐步骤 Ledoit–Wolf 密度评分、留一训练录像距离校准，再与执行/时长分数等权融合。没有调融合权重或重训 AEM。

这里采用平滑经验生存分数；不把它或 `exp(-score)` 称为有严格有限样本保证的共形概率。末段采样也不保证手已撤离。

## 版本校验与复现

```bash
git checkout v2026.10.27-tail-roi-evidence
python publication/verify_snapshot.py
```

[VERSION.json](VERSION.json) 固定主入口、源代码集合哈希和结果口径；[文件清单](publication/files.json) 固定每个发布文件的 SHA-256。源码保留实验时的原始字节，没有为发布修改计算逻辑。

这是**原研究工程的源码与证据快照**。原始数据、权重、逐帧缓存、SSH 工具和日志均不上传；完整运行仍需要原工程的路径和依赖。先读 [复现与依赖说明](publication/REPRODUCING.md)，不要把本仓库当作不需要数据的独立安装包，也不要直接运行历史 `run_pipeline.py`。

## 当前结论

五个任务都已参与过研究探索，不属于新的独立确认集。Pinwheels/Oatmeal 保留既有 checkpoint 与本地标签对应问题，未按测试标签重映射。单种子、有限正常验证录像及已有数据重复使用的限制仍然存在。

[G2 读出报告](g2-readout/report.md) 显示，尾帧 ROI 对状态/关系伪标签的实例读出也未可靠超过步骤均值，因此不能由 token 表现较差推出“脑补”或信息瓶颈。当前可保留的是末段观测的经验增量，不能宣称完整 CAVE、学习观察器或其机制已经验证成功。

原始 AEM 项目：<https://github.com/WenliangGuo/AEM>。第三方代码、模型和数据需按各自许可另行获取；本仓库不重新分发这些权重或数据。
