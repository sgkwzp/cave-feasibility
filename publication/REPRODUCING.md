# 复现范围与外部依赖

## 1. 本次发布保存什么

- `source/*.py`：原实验源码，逐字节保留。
- `deployment-mechanism/`：主版本冻结配置、对象查询、输入哈希、统计汇总和核验报告。
- `follow-up/` 与 `remaining-tasks/`：主版本的两任务和五任务 GT 前置阶段配置与证据。
- `g2-readout/source/` 及其报告：已完成的负面机制诊断，作为历史保留，不是主方法入口。
- 根目录的 `execution.json`、`report.md` 等记录属于最初的学习观察器阶段。当前状态以根目录 `README.md`、`VERSION.json` 和 `deployment-mechanism/report.md` 为准。

原报告中的逐录像观测、分数、bootstrap NPZ、图像 QA、模型及特征索引等引用指向原工程中保留的实验产物；未发布的内容不能仅靠 Git checkout 恢复。发布清单 `files.json` 是本次上传的精确边界。

## 2. 保持原工程目录

`source/common.py` 根据文件位置计算 `ROOT`、`PROJECT` 和兄弟目录 `OEV`。推荐将仓库放在原工程的相同层级：

```text
PROJECT/
  code/
    AEM/                           # 官方源码、metric.py、checkpoint 和发布辅助数据
    EgoPER/data/                   # 原有标注与划分
  data/data/<Task>/<task>_videos.zip
  ccfa-workfiles/experiments/
    cave-feasibility/              # 本仓库 ROOT
    oev-feasibility/               # 原观测缓存、检测器、本地依赖和前置结果
    aem-duration-complementarity/  # 五任务 AEM / duration 前置结果
    aem-segmentation-shift/        # 官方模型的预测游程与时间轴缓存
```

原工作区本身已满足这些路径关系；Git 初始化不会移动数据。新环境需要自行恢复有权限使用的原始数据与缓存。没有恢复前置产物时，不宣称可以一键重新得到所有结果。

### 主版本直接读取的资源

| 输入 | 用途 |
| --- | --- |
| `ROOT/manifest.json`, `ROOT/remaining-tasks/manifest.json` | 原五任务 GT 片段索引 |
| `ROOT/feature-index.json`, `ROOT/remaining-tasks/feature-index.json` | 已锁定帧索引、几何和旧缓存 |
| `ROOT/features`, `ROOT/roi`, `ROOT/embeddings` 及扩展阶段同名目录 | 同帧同查询锚点、特征与旧 ROI 数值复核 |
| `ROOT/follow-up/analysis/<task>/scores.json`, `ROOT/remaining-tasks/results/<task>/scores.json` | 固定 GT AEM/Duration/Post 对照 |
| `aem-segmentation-shift/cache/scores/<task>/<split>/*.npz` | 预测边界、类别、AEM 相似度、评测 GT 时间轴 |
| `aem-segmentation-shift/cache/<task>-metadata.json` | 官方指标历史锚点 |
| `oev-feasibility/cache/grounding-dino-tiny` | 冻结检测器及 processor |
| `oev-feasibility/cache/deps` | 原工程的本地 Python 依赖 |
| Hugging Face cache 中 `timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k` 的 `open_clip_model.safetensors` | 冻结视觉模型 |
| `ROOT/deployment-mechanism/observations`, `embeddings`, `metric-cache`, `cache` | 最新阶段的观测、评分、统计与 QA 产物 |

`deployment-mechanism/frozen-inputs.json` 和相关 signature 保存具体文件及模型哈希。缺失资源应明确恢复，不要删除哈希断言、用测试标签重映射或重新生成不同缓存冒充同一版本。

## 3. 环境

原记录确认：Linux、RTX 4090、Python 环境 `ED`、PyTorch `1.13.0+cu117`、`open_clip_torch` 模块版本 `2.32.0`。视觉特征以 fp16 缓存，下游使用 float32。服务器路径不是运行逻辑的一部分；无需复制 SSH 配置。

源码还导入 `numpy`, `scipy`, `scikit-learn`, `Pillow`, `opencv-python`, `transformers`，以及 AEM 的依赖。本次没有获取完整服务器包锁，也没有在全新环境重装验证，所以不发布虚构的完整 requirements 锁文件。保留能运行历史实验的环境，并以模型/输入哈希核验结果。尤其需匹配 Grounding DINO processor 的旧接口。

## 4. 入口与运行顺序

本次上传没有启动实验。以下命令供已经恢复原工程依赖、并准备好独立输出副本的环境使用；它们会写实验产物，应在新分支或工作副本中执行，保留冻结标签。

```bash
# 只校验已发布源码与证据；不训练、不联网、不改产物
python publication/verify_snapshot.py

# 在完整前置工程中：重建预测/前态观测清单，确认哈希后运行
python source/validate_prepare.py
python source/validate_observe.py --worker 0 --workers 1
python source/validate_score.py
python source/validate_analyze.py
```

前置阶段的固定 tail ROI 及 GT 分数必须已经存在。`validate_observe.py` 支持分片；所有分片完成后才执行评分。历史发布的 `validate_report.py` 会额外复核原运行的 **3 个 worker signature**、QA 图和全部缓存；单 worker 重跑不能不加区分地套用该历史审计条件。旧 `validate_finish.py` 是等待这些观测完成的 Linux 运行协调器，不是方法本体。

`validate_prepare.py` 会重写 manifest 与冻结记录；需要精确核验历史快照时，应先恢复并核对已记录的原输入，不能用新生成记录覆盖历史证据。历史 G1 学习观察器和 G2 读出不属于本轮默认执行链。

## 5. 审计解释

最新结果分两类：预测输入下全帧 AUROC；匹配 GT 输入下逐步骤宏 AUROC。官方 40 阈值网格 AUC、阈值平均 EDA 和精确 AUROC 也分别报告。不要把不同口径中的最大值组合成新的主表。

“留一录像”指 Gaussian 距离参考的校准；PCA 在正常训练集统一拟合。它不是整条表示/校准链都逐录像重拟合，也不是严格独立的共形校准。结果区间不包含正常参考重新拟合和跨种子变异。

后续新分析或新方法必须产生新 commit/tag；不要修改此标签所指源码后沿用原结果数字。
