# HyperCross 科学 HSI 主线：算法优化与创新计划

- 建立时间：2026-10-30
- 工作区：`G:\Datasets\HyperCross`（唯一写入位置）
- 数据：`G:\Datasets\HyperLeaf2024`、`G:\Datasets\PKG - HistologyHSI-GB`
- 设备：TCFM 环境 / RTX 5080 16 GB
- 车道声明：本目录下 `experiments/` 的一切结果默认是 **exploratory / development** 等级，
  只有再次通过 `benchmark/`（Claim Gate）复算的结果才能作为投稿证据。

## 1. 主线定位

保留**科学 HSI**（近距、小样本、非遥感）为主轴，两个任务：

1. **HyperLeaf2024**：整叶高光谱（204 波段 × 48 × 352），三目标回归
   （GrainWeight / Gsw / PhiPS2）+ 品种分类。
2. **HistologyHSI-GB**：13 名患者脑胶质瘤 ROI 二分类（rgb.png + 826 波段 ENVI 立方体）。

评测协议一律用**分组/域位移**协议：HyperLeaf 用 leave-one-variety-out（LOVO，4 折），
Histology 用 leave-one-patient-out（LOPO，13 折）。**禁止**随机样本切分。
一切拟合态（标准化、PCA、目标归一化、阈值、温度）只用训练折数据。

## 2. 当前诚实基线（来自 governed benchmark，作为对照锚点）

| 任务/协议 | 结果 | 来源 |
|---|---|---|
| Histology LOPO 3 seed（pooled OOF） | task-only-control Acc 0.8948 / BA 0.8714 / AUROC 0.8677 | `runs_v2/strict/histology/.../task-15.13-report.json`（strict_candidate，provenance 不全） |
| HyperLeaf 嵌套 LOVO 3 seed（pooled OOF R²） | train-fold-pca：GW 0.6227 / Gsw 0.0203 / Phi 0.4626 | `runs_v2/remediation/.../task-24.2-report.json`（primary_strict） |
| 同上，Gsw 最优分支 | rgb-pca-consistency-0.05：GW 0.5921 / Gsw 0.1013 / Phi 0.4737 | 同上 |
| 冻结胜出候选 | rgb-pca-consistency：GW 0.5070 / Gsw 0.0756 / Phi 0.4670 | 同上 |

**关键观察**
- 没有任何一个候选在三个目标上同时最优：GW/Phi 偏好 train-fold-pca，Gsw 偏好 consistency。
- 说明**存在"按目标路由 + 一致性正则"的联合空间尚未被利用**：把三者最优组合起来
  上界约为 GW 0.62 / Gsw 0.10 / Phi 0.47。
- 旧论文宣称的跨模态注意力在 `src/hypercross_routeA.py:118` 是**单 token 退化**的
  （softmax 恒为 1，Q/K 梯度为 0），必须显式证明或放弃。

## 3. 候选创新点（并行探索）

每条都预登记「假设 / 机制 / 预登记判据 / 需要的对照」。判据中的 delta 均以
同一 LOVO/LOPO 折、同一 seed 集比较，禁止跨协议 rank。

### A. 真正的 token 级跨模态对应（修复退化注意力）
- 假设：把每分支单 token 换成「空间 patch token × 波段组 token」，双向 cross-attention
  能带来超过纯 concat 的增益。
- 预登记判据：在相同预算下 `cross-attn` 相对 `concat` 与 `value-only`（去掉 softmax 的
  纯 value 投影控制）在 GW/Gsw/Phi 至少一项上 +0.02 R²，且三 seed 同号。
- 若判据不成立 → 记录为 negative evidence，从论文标题中移除 attention。

### B. 域/品种不变的谱表征（主线科学命题）
- 假设：LOVO 的本质是品种域位移；对谱分支加「品种对抗（GRL）+ 谱形状归一化
  （SNV/一阶、二阶导数）」能提升跨品种泛化。
- 预登记判据：held-out variety 的 pooled OOF R² 在 Gsw 上 +0.03，且 GW/Phi 不下降超过 0.01。
- 对照：无对抗、仅 SNV、仅导数。

### C. 物理先验谱特征 + 目标特定路由
- 假设：Gsw（气孔导度）与水分/红边bands 强相关，PhiPS2 与红边相关；显式构造
  连续统去除、导数、窄带比值指数，并按目标选择特征子集，可显著改善 Gsw。
- 预登记判据：Gsw R² ≥ 0.20（当前 0.10）且 GW ≥ 0.60。
- 对照：原始谱 Ridge/PLS（同协议）。

### D. 校准与选择性预测（可信科学分析，兼顾 AC 类期刊）
- 假设：Histology 的 fold-macro ECE 0.14、HyperLeaf 无校准；加温度/向量标定 +
  保形预测，可给出"带不确定性"的科学结论。
- 预登记判据：ECE 降到 ≤0.05 且覆盖率达名义水平 ±3%。
- 对照：未校准基线。

### E. 域泛化的测试时自适应（无标签）
- 假设：held-out variety/patient 上做 AdaBN/TENT 类无标签自适应，能恢复部分掉点。
- 预登记判据：相对无自适应 +0.03 R²（或 BA +0.02），且不使用任何目标标签。
- 风险：可能只是重新调了 BN 统计量，需要与「随机扰动对照」比较。

## 4. 实验队列（按序）

1. **W1（进行中）**：HyperLeaf LOVO 单 seed 筛选 —— 物理特征 / 域对抗 / 跨模态对照。
2. **W2**：对 W1 胜出者做 3 seed 复算 + bootstrap CI；Histology LOPO 同步跑同族想法。
3. **W3**：校准（D）与测试时自适应（E）。
4. **W4**：胜出候选回灌 `benchmark/` 复算，取得 `primary_strict` / `strict_candidate` 等级。

## 5. 投稿目标

- 顶会：ICML 2027（约 2027-01 截稿）为主，CVPR 2028 为备选；CVPR 2027 时间不可行。
- 期刊：Analytical Chemistry（AC）及 IF≈10 的适配期刊（如 TGRS / ISPRS J / Food Chemistry 方向），
  以「可信、可迁移的小样本 HSI 科学分析」为故事，物理先验 + 校准是它们的加分项。

## 6. 工程约束

- 只写 `G:\Datasets\HyperCross`；`src/`、`runs/` 只读。
- 删除中间 checkpoint 与缓存以节省空间（2026-10-30 已释放 ~25.6 GB），证据 JSON/MD 一律保留。
- 所有新结果写入 `runs_v3/`（新波次）或 `experiments/**/results/`，并同步更新 `docs/EXPERIMENT_LOG.md`。

## 7. 路线修正 v2（2026-11-06）

依据 W1–W6 证据：唯一可复现正向机制为「谱分支品种对抗（GRL λ≈1.0）」；
注意力、手搓物理特征、全融合对抗、后处理校准均为负结果（见 EXPERIMENT_LOG.md）。
主线故事改为「小样本科学 HSI 的域/品种不变谱表征与可迁移回归」，详见日志末节。

---

## 8. 路线修正 v3（2026-11-08，覆盖第 7 节）

第 7 节"唯一可复现正向机制为谱分支品种对抗（GRL λ≈1.0）"**已被 task-25.1 + W10 推翻**，详见
`docs/EXPERIMENT_LOG.md` 2026-11-08 节。要点：

1. 该增益是**表示条件性的**：只在"折内拟合逐波段 z-score"的谱表示下为正；
   在 governed 管线的"逐样本 SNV" 128 波段表示下，60 epoch Δ=-0.0143、200 epoch Δ=-0.1402（均三 seed 同负）。
   此前三次"复现"共享同一表示，属同一条件下的重复。
2. governed 管线的纯谱分支此前**欠训练**：同一 arm 从 60 epoch 的 Gsw -0.0200 升到 200 epoch 的 +0.1016。
3. 单分支纯谱 SNV-MLP（200 epoch、128 波段平均光谱）达到 **Gsw 0.1016 / PhiPS2 0.4781**，
   追平或超过 task 24.2 全部候选（其集合内**无纯谱候选**）。
   → "多模态一致性带来 Gsw/PhiPS2 增益"尚无证据；必须先建立同预算纯谱基线。
4. 当前最强、最稳健的机制是**表示与归一化设计**（逐样本 SNV + 足够预算）与**形态通路**，
   不是对抗对齐。对抗对齐的价值在于它是一个**可控的探针**：只改一个归一化开关即可双向翻转其符号，
   使"域-标签纠缠决定对齐符号"成为同数据集内可复现的可检验命题（这是顶会级主张的最可行入口）。

### 新主线的三条候选命题（按顶会可行性排序）

- **H1（首选）**：*当域是 nuisance 时对齐有益、当域与标签纠缠时对齐有害* —— 把观察升级为判据：
  在 inner domains 上量化每个目标的"域可解释方差"，用它**预测**哪个目标需要对冲；
  并实现"只对齐 nuisance 子空间"（GRL 仅作用于 label-predictive 子空间的正交补）。
  需要：合成对照 + 第三个数据集，把两数据集轶事升级为规律。
- **H2**：*小样本科学 HSI 的表示/归一化设计规律* —— 逐样本 SNV、导数、连续统去除等对跨品种迁移的
  系统影响（当前 SNV 单项 +0.16 R²，是本项目最大单点收益）。需要有控制的归一化网格与物理归因。
- **H3（AC/TGRS 类期刊的稳妥故事）**：小麦表型预测的可信跨品种迁移分析 ——
  以 LOVO 为主协议，报告 per-target/per-variety 分解、校准与不确定性、效率表，
  诚实呈现 Gsw 的困难与边界。

### 立即要做（工程优先级）

- **P0**：把纯谱 SNV 基线**纳入 governed 候选空间**并预登记同预算复算（含 60/200/400 epoch 曲线），
  这是当前最便宜且影响最大的一步；同时解释 24.2/24.3 的 epoch 策略（inner-OOF best-epoch≈4、
  MAX_SELECTION_EPOCH=80）是否系统性压制了简单模型。
- **P1**：H1 的判据化与正交补对齐实现（成本低，可直接复用现有 DANN 代码）。
- **P2**：第三个数据集 + 合成纠缠对照。

---

## 9. 路线修正 v4（2026-11-09）— 主线收敛到"表示设计"，并给出 governed 复算方案

W11/W12 之后，本项目的正向机制已经收敛到一个**单一、简单、可复现**的结论（详见 EXPERIMENT_LOG.md
2026-11-08/09 各节）：

| 模型（外折 LOVO，3 seed pooled OOF R²） | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| governed task-24.2 冻结胜出（`rgb-pca-consistency`） | 0.5070 | 0.0756 | 0.4670 |
| governed task-24.2 各目标最好值 | 0.6227 | 0.1013 | 0.4737 |
| **纯谱 `snv` MLP，200 epoch** | 0.6252 | **0.1075** | **0.4800** |
| **纯谱 `snv` MLP，400 epoch** | **0.6415** | 0.0958 | 0.4759 |

**结论**：一个只用 128/204 波段平均光谱、只做逐样本 SNV 的普通 MLP，在 Gsw 与 PhiPS2 上高于
governed task-24.2 的**全部**候选，在 GrainWeight 上打平/超过最好的候选。
而"多模态一致性"与"品种对抗对齐"分别属于**从未被检验**（24.2/24.3 候选空间里没有纯谱候选）
与**表示缺陷的症状**（一旦 SNV 去掉可移除的幅度 nuisance，对抗增益从 +0.171 掉到 +0.006）。

### 新的主线命题（paper）

> 小样本科学 HSI 的跨品种迁移，瓶颈在**谱表示的归一化/形状设计**，不在融合结构，
> 也不在域对抗对齐；并且"域对齐是否有益"可以**只用一个归一化开关**在同一数据集内双向复现，
> 其判据是"表示里是否还残留可移除的域 nuisance"。

三条可写的贡献：
1. 一个**纯谱基线即可超过多模态族**的诚实结论（含 governed 同预算 matched 对照）；
2. **对抗对齐的符号由表示决定**，并给出可检验判据（clean 模型弱 ⇔ Δ 大，见 W11 第 4 条）；
3. 一组**可复现的负结果**：跨模态注意力（含真 token + value-only 对照）、全融合对抗、
   逐目标表示路由、PCA 潜表示、MSC、导数/连续统去除、后处理校准。

### 立即执行：task 25.2（governed 复算，升级 `snv` 为正式候选）

- **候选集合**：`spectral-only::snv-mean-spectrum::mse`（新，一等候选）＋ task-24.2 的 5 个候选作为
  matched 对照，放进**同一** pipeline、同一 folds/seeds、同预算，并在预登记中固定
  `spectral_normalization_key = per-sample-snv`（**必须与 runner 实际输入一致**，见第 8 节的声明/实现不一致）。
- **预算曲线**：60 / 200 / 400 epoch 三点同时报告，验证排名在预算上的稳定性；
  并把"预算快照集成（200+400 平均）"作为预登记的第四点。
- **升级路径**：外折确认性通过后走 claim gate，目标等级 `strict_candidate`；official 测试预算保持 0。
- **必须同时修正**：24.2/24.3 的 `SCREEN_CAP=20` / `PROMOTION_CAP=80` 对晚收敛模型的系统性偏置，
  以及 24.2/24.3/24.4 结论需要 supersede 的说明。

### 之后的优先级（不变）

- **P1**：第三个数据集 + 合成纠缠对照，把"域-标签纠缠决定对齐符号"升级为标准；实现
  "只对齐 nuisance 子空间"（GRL 作用于 label-predictive 子空间的正交补）。
- **P2**：统计与基线补齐 —— ≥5 seeds、held-out-group bootstrap CI、per-class F1、per-variety 分解、
  参数量/FLOPs/显存/时延、label-fraction 曲线，以及 SpectralFormer / SSFTT / HybridSN / S2Mamba 与
  PLS/RF 经典谱基线在同协议下的复算。
- **口径纪律**：legacy `CCF_A_COMPARISON.md` 的 0.9189 / 0.4550（patient/random split）不得与
  governed LOVO/LOPO 数字并列。
