# HyperCross 项目现状说明与 CVPR 2027 投稿评估

- 生成日期：2026-11-10
- 受治理代码库（唯一写入工作区）：`G:\Datasets\HyperCross`
- 数据：`G:\Datasets\HyperLeaf2024`、`G:\Datasets\PKG - HistologyHSI-GB`
- 环境：TCFM 环境（Python 3.10.20 / torch 2.11.0+cu128）/ RTX 5080 16 GB
- 相关文档：`G:\Datasets\HyperCross\docs\MAINLINE_PLAN.md`、`docs\EXPERIMENT_LOG.md`（append-only，约 47.8 KB）
- 车道声明：本文件中的 governed 结果可作投稿证据；`experiments/wave*` 与 `runs_v3/` 的结果一律为 **exploratory / development**，必须经 `benchmark/`（Claim Gate）复算后才能升级。

---

## 0. 一句话结论（TL;DR）

1. **算法主线已经收敛到一个稳健、可复现、量级最大的正向机制**：*逐样本 SNV 谱表示 + 足够训练预算的纯谱 MLP*。跨品种 Gsw 从 **-0.1191** 抬到 **+0.1075**（+0.2266 R²），PhiPS2 从 0.4144 抬到 0.4800；该结论已在 **governed** 与 **exploratory** 两条独立代码路径上互相印证（Gsw 0.1016 vs 0.1075）。
2. **"多模态融合/一致性"与"域对抗对齐"在本项目证据里都不是正向机制**：前者在治理候选空间里*从未被公平检验*（24.2/24.3 候选集中没有纯谱候选）；后者是**表示缺陷的症状** —— 只改一个归一化开关（SNV 开/关），对抗增益就从 +0.1714 掉到 +0.0062。
3. **但在 6 天内完成一篇 CVPR 主赛道投稿不可行**，原因不是"写不完"，而是**证据结构不支撑 CVPR 的主张**：缺现代基线同协议复算、≥5 seed、held-out-group bootstrap CI、效率表、第三个数据集，而所有"融合/对齐/路由/校准"类创新点都已被本项目自己的消融否定。详见 §1 与 §5。

---

## 1. 关键时间线（硬约束）

| 节点 | 日期 | 距今 |
|---|---|---|
| **CVPR 2027 论文提交（AOE）** | **2026-11-16** | **6 天** |
| CVPR 2027 作者名单锁定 / 注册 | 2026-11-10 | 今天 |
| ICML 2027 论文提交（约） | 2027-01 下旬 | 约 10 周 |
| ICCV 2027 论文提交（约） | 2027-03 上旬 | 约 17 周 |
| CVPR 2028 论文提交（约） | 2027-11 中旬 | 约 12 个月 |
| 期刊路线（AC / TGRS / ISPRS J / Food Chemistry，IF≈10） | 随时可投 | 立即 |

判定：

- **CVPR 2027 主赛道：不建议强行投稿。** CVPR 的 HSI 相关工作已标配"多场景 + 严格空间/分组划分 + 现代基线（SpectralFormer / SSFTT / HybridSN / S2Mamba）+ ≥5 seed / bootstrap"。当前项目缺这些，且核心卖点（跨模态注意力、域对抗、逐目标路由、后处理校准）**全部被自证无效**。强行投递最可能的结局是 Reject，并浪费一次窗口。
- **现实可行的会议目标**：① **ICCV 2027**（CV 领域 A*，约 17 周，足以把"表示设计规律"做成扎实的规律型论文）；② **ICML 2027**（约 10 周，偏好"机制/规律 + 可控实验"，与本项目已有的"用一个开关双向翻转现象"的证据形态高度契合）；③ **CVPR 2028**（约 12 个月，做完整 benchmark 版本）。
- **最快的确定性产出**：立刻用现有证据投 **Applied Catalysis / TGRS / ISPRS J / Food Chemistry 类 IF≈10 期刊**（更看重科学发现与诚实评估，不强制大规模 baseline 矩阵）。建议与会议线并行推进，不互相阻塞。

---

## 2. 资产总览

### 2.1 数据

| 数据集 | 位置 | 规模 | 用途 | 状态 |
|---|---|---|---|---|
| HyperLeaf2024 | `G:\Datasets\HyperLeaf2024` | 1590 样本；204 波段 × 48 × 352 立方体 + 3×32×64 RGB；4 品种（Heerup / Kvium / Rembrandt / Sheriff）；3 个回归目标（GrainWeight / Gsw / PhiPS2） | **主线** | 在用 |
| HistologyHSI-GB | `G:\Datasets\PKG - HistologyHSI-GB` | 469 ROI / 13 患者；826 波段 ENVI 立方体 + rgb.png；二分类 | 第二域（分类） | 在用 |
| Indian Pines / Salinas / Pavia / Houston2013 | 已从 `G:` 移除（旧代码遗留在 `F:\ZJU\科研\Hyperspectral\论文\HyperCross\`） | — | 遥感 sanity track | **已放弃**（见 §7） |

**协议纪律**：HyperLeaf 一律 leave-one-variety-out（LOVO，4 折），Histology 一律 leave-one-patient-out（LOPO，13 折）。**禁止随机样本切分**。所有拟合态（标准化、PCA、目标归一化、阈值、温度）只用训练折数据。seed 固定集合：**7 / 42 / 123**。

### 2.2 代码（三层）

**(a) 受治理 benchmark —— `G:\Datasets\HyperCross\benchmark\`（约 50 个模块，投稿证据层）**

| 文件 | 作用 |
|---|---|
| `benchmark\schema\*` | 全部 Canonical_* schema（protocol / prediction / selection / aggregation / metrics…） |
| `benchmark\catalog\*` | Dataset Catalog、Sample Manifest、raw cube 读取、de-identification |
| `benchmark\protocols\hyperleaf_lovo.py` | `leaf.lovo.nested.v1`（外层 4 折 LOVO + 内层品种选择） |
| `benchmark\protocols\histology_lopo_fixed.py` / `histology_nested.py` | `hist.lopo-fixed.v1` / `hist.patient-nested5.v1` |
| `benchmark\engine\training.py` / `evaluation.py` / `selection.py` / `orchestration.py` | 训练、one-shot 外层评估（evaluation lease）、选择冻结、可恢复 stage machine |
| `benchmark\metrics\*` | prediction-first 指标、patient bootstrap、paired comparison、aggregate builder |
| `benchmark\registry\*` / `recovery\*` | append-only run registry、supersession、recovery manager |
| `benchmark\transforms\preprocessing.py`（38.8 KB） | fold-scoped 预处理；**注意 L397–400 的实际输入是逐样本 SNV `((raw-mean)/std)`，与预登记声明 `per-band-zscore-inner-train.v1` 不一致（待修）** |
| `benchmark\models\hyperleaf_nuisance_alignment.py`（11.9 KB） | `HyperLeafSpectralOnly`、`HyperLeafSpectralVarietyAdversarial`、λ0.3 子类、routed 变体、`SPECTRUM_ONLY_INPUT_CONTRACT`、registry |
| `benchmark\runners\hyperleaf_nuisance_alignment.py`（35.4 KB / 842 行） | **task-25.1 runner**：`--freeze/--execute/--resume/--epochs/--seeds/--arms/--smoke/--fold-limit`，arm registry、loss/predict/evaluate、预登记判据裁决、markdown 输出 |
| `benchmark\runners\hyperleaf_optimization_registry.py`（33.3 KB） | `_build_candidates`（L377–430）= 候选空间定义；**这里是"没有纯谱候选"的根源** |
| `benchmark\runners\hyperleaf_screening.py`（58.5 KB）/ `hyperleaf_strict.py`（50.4 KB）/ `hyperleaf_smoke.py`（33.7 KB） | 筛选 / 严格 / 冒烟 runner |

**(b) 探索波次 —— `G:\Datasets\HyperCross\experiments\wave1..wave12\*.py`（12 个脚本 → `runs_v3\wave*\`）**

| 波次 | 脚本 | 主题 |
|---|---|---|
| W1 | `wave1\hyperleaf_lovo_screen.py` | HyperLeaf LOVO 筛选（物理特征 / 跨模态注意力对照 / 对抗） |
| W2 | `wave2\hyperleaf_lovo_dualview.py` | 双视图一致性（负结果） |
| W3 | `wave3\histology_lopo_screen.py` | Histology LOPO 筛选 |
| W4 / W5 | `wave4\histology_calibration.py` / `wave5\governed_histology_calibration.py` | 温度/Platt 校准（负结果） |
| W6 | `wave6\hyperleaf_dann_sweep.py` | 对抗强度扫描（λ=0.1/0.3/1.0/2.0/3.0） |
| W7 | `wave7\hyperleaf_routed_adversarial.py` | 按目标路由 + 集成 |
| W8 | `wave8\histology_patient_dann.py` | 患者对抗（有害） |
| W9 | `wave9\hyperleaf_instrument_proxy_study.py` | 治理选择仪器有效性研究 |
| W10 | `wave10\hyperleaf_representation_sensitivity.py` | 归一化敏感性网格（4 变体 × clean/DANN） |
| W11 | `wave11\hyperleaf_spectral_representation_grid.py` | 谱表示网格（8 表示 × clean/DANN + 域可识别性诊断） |
| W12 | `wave12\hyperleaf_representation_routing.py` | 修正 MSC、新增化学计量潜表示、诚实的逐目标表示路由（负结果）+ 预算曲线 |

**缓存（使全部波次可秒级复跑）**：`experiments\wave1\cache_leaf_spec_rgb.npz`（1590 × 204 平均光谱 + RGB），`experiments\wave1\cache_leaf_cube_small.npy`（654 MB）。

**(c) 遗留代码 —— `F:\ZJU\科研\Hyperspectral\论文\HyperCross\`（约 100 个脚本 + 旧论文模型 `src\hypercross_routeA.py` 40 KB）**
早期探索（2026-10-07 ～ 10-19），含遥感场景（Indian Pines / Salinas / Pavia / Houston）、S2Mamba 对照、token-cross、伪标签等。**其数字（如 0.9189 / 0.4550）使用的是 patient/random split，严禁与 governed LOVO/LOPO 数字并列。**

### 2.3 结果产物

- `G:\Datasets\HyperCross\runs_v2\`（governed）：`strict\hyperleaf-nuisance-alignment\...`（task-25.1）、`remediation\...task-24-2-...`（task-24.2, `primary_strict`）、`optimization\hyperleaf\...latest-task-24.4-summary.json`（`all_contexts_fell_back`）、`screening\{histology,hyperleaf}`（14.12）、`strict\{histology,hyperleaf}`（15.13 / 14.13）
- `G:\Datasets\HyperCross\runs_v3\`（exploratory）：`wave1` … `wave12` 的 `*_summary.{json,md}`
- `G:\Datasets\HyperCross\configs\optimization\v2\hyperleaf-task-25-preregistration.json`（冻结，hash `69783c2d…c781e8`）
---

## 3. 当前结果全景

### 3.1 Governed 锚点（可作投稿证据层）

| 任务 / 协议 | 结果 | 等级 |
|---|---|---|
| HistologyHSI-GB，LOPO 13 患者，3 seed | `local-route-a.task-only-control.v1`：Acc **0.8948** / BA **0.8714** / AUROC 0.8677 | `strict_candidate`（provenance 不全） |
| HyperLeaf2024，嵌套 LOVO，3 seed，pooled OOF R² | task-24.2 胜出 `rgb-pca-consistency`：GW 0.5070 / Gsw 0.0756 / Phi 0.4670 | `primary_strict`，`promotion_permitted=true` |
| 同上，各目标最好候选（上界参考） | `train-fold-pca`：GW **0.6227** / Gsw 0.0203 / Phi 0.4626；`rgb-pca-consistency-0.05`：GW 0.5921 / Gsw **0.1013** / Phi **0.4737** | 同上 |
| HyperLeaf 官方 test（820 样本） | **未消耗**（`official_evaluation_budget_consumed=0`） | sealed |
| task-25.1 预登记判据（60 epoch，6 arms） | **C1 NO / C2 NO / C3 NO / C4 YES** | `development` |

> C4 = YES 是关键：在**同一 runner**内，`rgb-pca-consistency-010` 复现出 Gsw **+0.0961**、`train-fold-pca` **+0.0617**，落在 task-24.2 的 0.1013 / 0.0756 区间内。⇒ **说明 harness 口径无误，C1 的失败是机制失败，不是 bug。**

### 3.2 探索波次结果（exploratory tier，外折 LOVO LOVO 全部 3 seed 均值）

**W1 HyperLeaf LOVO 筛选（200 ep，pooled OOF R²）**

| candidate | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `dann-spec` | 0.6442 | -0.0200 | 0.4366 |
| `mlp-spec`（= 折内逐波段 z） | 0.6375 | -0.1191 | 0.4144 |
| `fusion-valueonly` | 0.5861 | -0.1425 | 0.4159 |
| `fusion-crossattn` | 0.5743 | -0.1529 | 0.4223 |
| `fusion-concat` | 0.5759 | -0.1628 | 0.4104 |
| `ridge-spec` | 0.3439 | -0.1654 | 0.4800 |
| `mlp-phys`（手工物理特征） | 0.4851 | -0.2306 | 0.5074 |
| `cnn-rgb` | -0.0387 | -0.4339 | -0.0564 |
| `ridge-phys` | 0.0794 | -1.2088 | 0.3143 |

⇒ **真正的 token 级 cross-attention 与 matched `fusion-valueonly` 对照无差异**；手工物理特征更差。

**W2 双视图一致性**：`dual-consistency` GW 0.0737 / Gsw -0.3427 / Phi 0.0090（**崩塌**）；`pca-cnn` GW -0.0126。⇒ 多模态一致性在纯谱基线面前毫无优势。

**W3 Histology LOPO（120 ep，469 ROI / 13 患者）**

| candidate | Acc | BA | AUROC | ECE |
|---|---:|---:|---:|---:|
| `mlp-spec` | 0.8472 | 0.8286 | 0.8606 | 0.1267 |
| `fusion-concat` | 0.8152 | 0.8229 | 0.9190 | 0.1658 |
| `fusion-valueonly` | 0.8173 | 0.8200 | 0.9125 | 0.1557 |
| `dann-spec` | 0.8088 | 0.8094 | 0.8935 | 0.1008 |
| `cnn-rgb` | 0.7989 | 0.8071 | 0.9155 | 0.1805 |
| `fusion-crossattn` | 0.7918 | 0.7993 | 0.9053 | 0.1951 |

Governed 参考（task 15.13 pooled OOF）：Acc 0.8948 / BA **0.8714** / AUROC 0.8677 ⇒ 探索结果**全部低于** governed 锚点。

**W4 / W5 校准**：温度标定把 ECE 从 0.13～0.20 压到 0.035～0.073，但 **BA 一律下降**（`fusion-concat` BA 0.8229 → 0.8250 持平，`fusion-crossattn` 0.7993 → 0.7679 变差）。⇒ 校准是"可信度"议题，不是提分手段。

**W6 对抗扫描（200 ep）**：`dann-spec-lam1.0` GW 0.6057 / Gsw 0.0523 / Phi 0.4308（表面最好）；但 `dann-fusion-*` 全系崩塌（GW -0.09 ～ -0.48）。**当时代码使用折内逐波段 z 表示 —— 这正是 §3.4 要否定的条件。**

**W7 路由 / 集成**：`routed` GW 0.6504 / Gsw 0.0523 / Phi 0.4149；`ensemble` GW 0.6468 / Gsw -0.0051。⇒ 相对 `clean`(0.6375 / -0.1191) 的"改进"来自表示（fold-z + 对抗），不是路由本身。

**W8 患者对抗（Histology）**：`dann-spec-1.0` BA 0.7909（vs clean 0.8286）、`fusion-dann-all-1.0` BA **0.4590**。⇒ 患者与标签纠缠 ⇒ 对齐有害。（与 §3.4 的"两端口同一机制"呼应。）

**W9 治理选择仪器有效性**：in-domain（2 训练品种）难度系统性高于外折难度，其 Δ 与外折 Δ 符号甚至相反（inner-stratified Gsw Δ = **-0.0568** vs outer-lovo **+0.1714**）。⇒ **governed 的 inner 选择仪器本身不可靠**，这解释了 task-24.4"所有上下文都 fallback"。

### 3.3 核心正向结论：逐样本 SNV 谱表示（W10 / W11 / W12）

W10（60 ep）与 W11（200 ep）在同一网格上只切换归一化，直接给出**符号翻转**：

| 表示 | 波段 | clean GW | clean Gsw | clean Phi | Δ(DANN−clean) Gsw |
|---|---:|---:|---:|---:|---:|
| `fold-z-raw`（折内逐波段 z，保幅度） | 204 | 0.6375 | **-0.1191** | 0.4144 | **+0.1714** |
| **`snv-raw`（逐样本 SNV）** | 204 | 0.6252 | **+0.1075** | **0.4800** | **+0.0062** |
| `snv-z`（SNV 后再逐特征 z） | 204 | 0.6209 | -0.0531 | 0.4657 | +0.0817 |
| `snv-d1-z`（+一阶导） | 408 | 0.5714 | -0.1644 | 0.4875 | +0.0845 |
| `snv-cr-z`（+连续统去除） | 408 | 0.4982 | -0.1651 | 0.4535 | +0.1435 |
| `snv-d1-cr-z` | 612 | 0.5027 | -0.1858 | **0.5020** | +0.1089 |
| `msc-z` | 204 | 0.6055 | -0.0821 | 0.4550 | +0.0781 |

W12（200 ep，单表示 + 3 seed std；MSC 已修正为作用于原始光谱）：

| 表示 | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `fold-z` | 0.6375 ±0.0058 | -0.1191 ±0.0142 | 0.4144 ±0.0085 |
| **`snv`（200 ep）** | 0.6252 ±0.0009 | **+0.1075 ±0.0014** | **0.4800 ±0.0001** |
| **`snv`（400 ep）** | **0.6415 ±0.0007** | +0.0958 ±0.0016 | 0.4759 ±0.0005 |
| `snv-pca32` | 0.2557 ±0.0839 | -0.3088 ±0.0372 | 0.4284 ±0.0292 |
| `msc`（对原始光谱） | 0.6034 ±0.0028 | -0.0843 ±0.0186 | 0.4539 ±0.0010 |
| `snv-d1-cr-z` | 0.5027 ±0.0277 | -0.1858 ±0.0298 | 0.5020 ±0.0052 |

**四条可写进论文的结论**

1. **逐样本 SNV 是本项目迄今最大的单点算法收益**：Gsw **+0.2266**、PhiPS2 **+0.0656**（相对 `fold-z-raw`），且三 seed 极稳（std ≤ 0.0016）。
2. **一个纯谱单分支模型即超过整个多模态候选族**：`snv`+普通 MLP（200 ep）= GW 0.6252 / Gsw 0.1075 / Phi 0.4800；在 Gsw、PhiPS2 上高于 task-24.2 **全部**候选（最高 0.1013 / 0.4737），GW 与最好的 `train-fold-pca`（0.6227）打平；400 ep 版 GW **0.6415** 反超全部候选。
3. **导数 / 连续统去除 / SNV 后再做逐特征 z 都变差**；`snv-pca32` 甚至 GW 崩塌到 0.2557。⇒ 信息在**平滑的 SNV 光谱形状**里，微分与连续统去除放大 nuisance。
4. **跨代码路径一致性（关键）**：

| 路径 | 表示 | 波段 | 编码器 / 优化 | Gsw | PhiPS2 | GrainWeight |
|---|---|---:|---|---:|---:|---:|
| governed task-25.1 `control::spectral-only`（200 ep） | 逐样本 SNV | 128 | `_SpectrumEncoder(64)`，AdamW lr 2e-4，bs 64 | **+0.1016** | **0.4781** | 0.5230 |
| 探索 wave-12 `snv`（200 ep） | 逐样本 SNV | 204 | MLP(256)，AdamW lr 1e-3，bs 128 | **+0.1075** | **0.4800** | 0.6252 |

⇒ Gsw 与 PhiPS2 在两条路径上差 0.006 / 0.002，**跨波段数、跨编码器、跨学习率稳健**；唯一分歧是 GrainWeight（0.523 vs 0.625），指向编码器/优化配置而非表示本身。

### 3.4 机制判定表（"什么成立 / 什么不成立"）

| 假设机制 | 判定 | 关键证据 |
|---|---|---|
| 谱表示设计（逐样本 SNV） | **成立（强）** | +0.2266 Gsw；两条独立代码路径复现；3 seed std ≤0.0016 |
| 跨模态一致性 / 双视图融合 | **不成立** | W2 全面崩塌（GW 0.0737 / Gsw -0.3427）；task-24.2 的多模态候选在 Gsw/Phi 上低于纯谱基线 |
| 真 token 级 cross-attention | **不成立** | W1 `fusion-crossattn` 与 matched `fusion-valueonly` 无差异；旧论文注意力在 `src\hypercross_routeA.py:122-123,138,145,157` 是**单 token 退化**（softmax≡1，Q/K 梯度为 0） |
| 品种/域对抗对齐（DANN / GRL） | **表示条件性成立，本质是"可移除 nuisance 的回填"** | 同一网格只换归一化：`fold-z` Δ=+0.1714 ↔ `snv` Δ=+0.0062；governed 60 ep Δ=-0.0143、200 ep Δ=-0.1402（三 seed 同负） |
| 手工物理特征（连续统去除 / 窄带比值） | **不成立** | W1 `mlp-phys` Gsw -0.2306、`ridge-phys` Gsw -1.2088 |
| 后处理校准（温度 / Platt） | **不成立（仅改善 ECE）** | W4/W5：ECE 0.13→0.06，但 BA 不变或下降 |
| 逐目标"表示路由" | **不成立（诚实负结果）** | W12：内层选择的 routed 策略 GW 0.6046 / Gsw 0.1075 / Phi 0.4442，**劣于一律用 `snv`**（0.6252 / 0.1075 / 0.4800）；外层 oracle 上界 0.6375 / 0.1075 / 0.5020 |
| 全融合 DANN | **不成立（灾难性）** | W6 `dann-fusion-lam1.0` GW -0.4822 / Gsw -0.0754 |
| 患者对抗（Histology） | **有害** | W8 `dann-spec-1.0` BA 0.7909 vs clean 0.8286；`fusion-dann-all-1.0` BA 0.4590 |
| 全域不变性（across domain，两面同一机制） | **成立（可解释）** | Histology：患者与标签纠缠 ⇒ 对齐有害；HyperLeaf：品种基本是 nuisance ⇒（在保幅度表示下）对齐暂时有益 ⇒ **"域−标签纠缠度决定对齐符号"** |

**可检验判据（W9/W11 发现）**：Δ(DANN−clean) ≈ 正比于"clean 模型有多弱" ——
`fold-z-raw` clean -0.119 → Δ +0.1714；`snv-cr-z` clean -0.165 → Δ +0.1435；`snv-d1-z` clean -0.164 → Δ +0.0845；`snv-raw` clean **+0.108** → Δ **+0.0062**。
**"表示里只要还留着可移除的水位/幅度 nuisance，对齐就显得有用；一旦表示把它去掉，增益就消失。"** 正确的做法是**改表示，而不是加对抗目标** —— 这也解释了治理管线（保幅度的谱表示）为何会"发现"对抗有效。
---

## 4. 论文故事（当前证据支持的版本）

**核心命题（一句话）**

> 在小样本科学高光谱的跨品种迁移中，**瓶颈是谱表示的归一化/形状设计，而不是融合结构，也不是域对抗对齐**；并且"域对齐是否有益"可以由**表示里是否残留可移除的域 nuisance** 决定 —— 这一点可以在**同一数据集内只用一次归一化开关（SNV 开/关）双向复现**。

**三条贡献**

1. **一个诚实的纯谱基线即超过整族多模态候选**（含 governed 同预算 matched 对照）。这直接挑战"科学 HSI 需要多模态融合"的默认假设。
2. **域对齐的符号由表示决定**，并给出可检验判据："clean 模型弱 ⇔ Δ 大"，且在 Histology（纠缠 ⇒ 有害）与 HyperLeaf（nuisance ⇒ 暂时有益）两端形成一致解释。
3. **一组可复现的负结果**：跨模态注意力（含真 token + value-only 对照）、全融合对抗、逐目标表示路由、PCA 潜表示、MSC、导数/连续统去除、后处理校准。负结果对审稿人是加分项，但**必须放在"我们做对了什么"之后**。

**当前不建议的卖点**（易被审稿人一击必杀）

- "跨模态注意力/融合带来增益" —— 无证据，且旧实现的注意力是退化的。
- "域对抗提升跨域泛化" —— 只在保幅度表示下成立，是症状不是机制。
- "SOTA on HyperLeaf/Histology" —— 缺少同协议现代基线与统计检验，无法宣称 SOTA。

---

## 5. 距 CVPR/ICML 主赛道的差距清单（Gap List）

按"审稿人会直接要求"的顺序：

**G1. 现代基线缺失（最致命）**
需在同一 LOVO/LOPO 协议、同 seed、同预算下复算：`SpectralFormer`、`SSFTT`、`HybridSN`、`S2Mamba/HSIMamba`、masked-pretraining 基线，以及经典谱基线 `PLS` / `RF` / `Ridge`。
现状：`external\spectralformer\` 只有源码（`vit_pytorch.py` + `einops`），**未在 governed 协议下跑过**；`src\modern_hsi_models.py`（2.6 KB）过小，不构成复现。
备注：W1 的 `ridge-spec` 已在 fold-z 表示下拿到 PhiPS2 **0.4800**，与最强神经结果持平 —— 这说明**经典基线必须做**，否则审稿人会用 Ridge/PLS 打掉"深度学习必要性"。

**G2. 统计强度不足**
现状 3 seed；需 ≥5 seed + held-out-group bootstrap CI + per-class F1 / per-variety 分解 + paired comparison。
（`benchmark\metrics\statistics.py` 与 `10.4 patient bootstrap / paired comparison` 已实现，缺的是**跑够 seed**。）

**G3. 数据集覆盖与"规律 vs 轶事"**
现状 2 个科学域（1 回归 + 1 分类）+ 已放弃的遥感 track。要把"域−标签纠缠决定对齐符号"升级为**规律**，需要：
① 第三个科学 HSI 数据集；② **合成纠缠对照**（可连续调 entanglement 强度，观察 Δ 符号翻转）。

**G4. 效率与可复现性**
缺参数量 / FLOPs / 峰值显存 / 训练时间 / 单样本时延表；缺 label-fraction 曲线；缺统一复现脚本。

**G5. 治理层的研究级缺口（`.kiro\specs\cvpr2027-hyperspectral-benchmark\tasks.md`）**
未完成任务：**24.4、24.5、24.6、24.7、24.8、24.9**，以及 14.2（one-shot official，预算 0）、测试任务 14.6 / 14.7 / 14.8、17、18。
其中 **24.7–24.9（Histology lineage-complete 重跑）** 是 Histology 侧维持 `primary_strict` 的前提（当前 15.13 的 `provenance 不全`）。

**G6. 口径一致性（内部必做）**
① 修 `per-band-zscore-inner-train.v1` 的声明/实现不一致（`benchmark\transforms\preprocessing.py` L397–400）；
② 解释 governed GW 0.5230 与探索 GW 0.6252 的差异（编码器/优化配置）；
③ 说明 `SCREEN_CAP=20` / `PROMOTION_CAP=80` 对晚收敛简单模型的系统性偏置，并 supersede 24.2/24.3/24.4 的结论。

---

## 6. 下一步计划（按优先级）

### P0 —— governed 复算：task-25.2（1～2 天，最便宜、影响最大）

把纯谱 SNV 基线**升级为正式候选**，在 governed 管线内预登记复算：

- **候选集**：`spectral-only::snv-mean-spectrum::mse`（新，一等候选）+ task-24.2 的 5 个候选作为 **matched 对照**，同一 pipeline / 同一 folds / 同一 seeds / 同预算。
- **预登记**：固定 `spectral_normalization_key = per-sample-snv`（**必须与 runner 实际输入一致**，见 G6①）。
- **预算曲线**：60 / 200 / 400 epoch 三点同时报告，验证排名在预算上的稳定性；第四点 = 预登记的"预算快照集成（200+400 预测平均）"。
- **升级路径**：外折确认性通过后走 Claim Gate，目标等级 `strict_candidate`；**官方测试预算保持 0**。
- **必须同时完成**：修正 SCREEN_CAP/PROMOTION_CAP 偏置；supersede 24.2 / 24.3 / 24.4 的结论。

### P1 —— 把"机制"做成"规律"（3～6 周）

- 实现 **"只对齐 nuisance 子空间"**：GRL 仅作用于 label-predictive 子空间的正交补（复用现有 DANN 代码，成本低）；提出并验证 **domain-explainable-variance** 指标来*预测*哪个目标需要对齐。
- 合成纠缠对照（可调 entanglement），把观察升级为判据。

### P2 —— 统计与基线补齐（4～8 周，投稿前必做）

- ≥5 seeds；held-out-group bootstrap CI；per-class F1；per-variety 分解；效率表（参数量/FLOPs/显存/训练时间/时延）；label-fraction 曲线。
- 同协议复算：SpectralFormer / SSFTT / HybridSN / S2Mamba + PLS / RF / Ridge。
- Histology 侧完成 lineage-complete 重跑（24.7–24.9），恢复 `primary_strict`。

### 建议的排期（两条线并行）

| 周次 | 会议线（ICCV 2027 / ICML 2027） | 期刊线（IF≈10） |
|---|---|---|
| 第 1 周 | P0：task-25.2 governed 复算 + 口径修正 | 用现有结果起草 Short Communication / 全文骨架 |
| 第 2–3 周 | P1：正交补对齐 + domain-variance 判据 | 补 per-variety / 校准 / 效率表 |
| 第 4–6 周 | P2：≥5 seed + bootstrap CI + 现代基线 | 第三数据集或合成对照（视投入） |
| 第 7–9 周 | 写作 / 内部审阅 / rebuttal 预演 | 投稿 |
| 约 17 周 | ICCV 2027 投稿（或 ICML 2027，约 10 周） | — |

---

## 7. 风险与口径纪律

**风险**

| 风险 | 说明 | 缓解 |
|---|---|---|
| Gsw 绝对水平低 | 最强 Gsw R² ≈ 0.11，绝对值偏低，审稿人会问"实用价值" | 明确表述为"跨品种迁移的困难被如实暴露"，并把 Gsw 作为 negative/boundary case；期刊线强调诚实评估 |
| 单数据集机制 | "对齐符号由表示决定"目前只在 2 域 + 1 数据集内成立 | P1 的合成对照 + 第三数据集 |
| 搜索空间偏置 | 24.2/24.3 的 20/80 epoch 上限系统性压制简单模型 → 历史结论可能被推翻 | P0 的 60/200/400 预算曲线 + supersede 声明 |
| 口径混用 | legacy `CCF_A_COMPARISON.md`（0.9189 / 0.4550，patient/random split）与 governed LOVO/LOPO 不可比 | **严禁并列**；文档中一律标注协议 |
| 治理层 provenance | 15.13 标注 `provenance 不全` | 24.7–24.9 重跑 |

**口径纪律（写进任何稿件前的 checklist）**

1. 只引用 `runs_v2/`（governed）中的数字作为主结果；`runs_v3/` 必须标注 exploratory。
2. 任何比较必须绑定相同的 data version / protocol manifest / representation / selection rule / seed set / metric。
3. 禁止把 LOVO/LOPO 数字与随机切分数字（legacy）放在同一张表。
4. 官方 test（820 样本）预算不消耗，直到最终一次 one-shot（14.2）。

---

## 8. 附录：关键文件索引与复现入口

**文档**
- `G:\Datasets\HyperCross\docs\MAINLINE_PLAN.md` —— 主线计划（§9 为最新；§7/§8 已被取代）
- `G:\Datasets\HyperCross\docs\EXPERIMENT_LOG.md` —— append-only 实验日志（12 波次 + 3 次代码审计）
- 本文件（`F:\ZJU\科研\Hyperspectral\论文\STATUS_FOR_CVPR2027_20261110.md`，副本见 `G:\Datasets\HyperCross\docs\`）

**预登记**
- `G:\Datasets\HyperCross\configs\optimization\v2\hyperleaf-task-25-preregistration.json`（hash `69783c2d5c392a41f1ad1ffa04963eb7f53b620e6ffa4c27b0f350826ce781e8`）

**复现入口（在 TCFM 环境下、Windows PowerShell）**
- HyperLeaf 表示网格（W11）：`python experiments\wave11\hyperleaf_spectral_representation_grid.py`
- HyperLeaf 表示路由 / 预算（W12）：`python experiments\wave12\hyperleaf_representation_routing.py`
- task-25.1 复跑：`python -m benchmark.runners.hyperleaf_nuisance_alignment --execute --epochs 200 --seeds 7 42 123 --arms control::spectral-only::mse,arm::spectral-dann-lam1.0`
- 缓存：`experiments\wave1\cache_leaf_spec_rgb.npz`（1590×204 平均光谱 + RGB，使全部波次秒级复跑）

**关键结果文件**
- `runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\task-24.2-report.json`（`primary_strict`）
- `runs_v2\strict\hyperleaf-nuisance-alignment\v1\deterministic-float32\task-25-1-20261108T130712621781Z\task-25.1-summary.md`（60 ep，C1–C4）
- `runs_v2\strict\hyperleaf-nuisance-alignment\v1\smoke\task-25-1-20261108T141414395253Z\task-25.1-summary.md`（200 ep 诊断）
- `runs_v3\wave11\spectral-representation-grid_summary.md`（谱表示网格）
- `runs_v3\wave12\representation-routing_summary.md` / `snv-budget400_summary.md`（路由 + 预算）
- `runs_v2\strict\histology\...\task-15.13-report.json`（Histology LOPO 锚点）

---

## 附：判定汇总（一页版）

| 问题 | 回答 |
|---|---|
| 算法主线是什么？ | 逐样本 SNV 谱表示 + 纯谱 MLP；**表示设计 > 融合结构 > 域对齐** |
| 最强的正向数字？ | Gsw +0.1075（相对 fold-z **+0.2266**）、PhiPS2 0.4800、GW 0.6415（400 ep） |
| 可以宣称"超过多模态族"吗？ | 可以（在 governed 同预算对照补齐后），当前为 exploratory |
| 可以宣称"域对抗有效"吗？ | **不可以**，是表示缺陷的症状 |
| 现在能投 CVPR 2027 吗？ | **不建议**（6 天 + 证据结构不足） |
| 最现实的会议目标？ | ICCV 2027（约 17 周）或 ICML 2027（约 10 周） |
| 最快的确定性产出？ | 期刊线（AC / TGRS / ISPRS J，IF≈10），可立即启动 |
| 第一步做什么？ | task-25.2 governed 复算 + 修声明/实现口径不一致 |
