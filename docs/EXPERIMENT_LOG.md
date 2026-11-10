# 实验日志（HyperCross 科学 HSI 主线）

所有条目按时间倒序追加。等级标注沿用 `benchmark/reporting/claim_gate.py` 的语义：
`exploratory` < `development` < `strict_candidate` < `primary_strict`。

---

## 2026-10-30 — 侦察、清理、W1 启动

### 侦察结论
- 主线两任务确认为 HyperLeaf2024（LOVO）与 HistologyHSI-GB（LOPO）；遥感数据不在 `G:\Datasets`，
  按需求不参与主线。
- 历史证据锚点（不重跑，仅引用）：
  - Histology LOPO 3 seed pooled OOF：`local-route-a.task-only-control.v1` Acc 0.8948 / BA 0.8714 / AUROC 0.8677
    （`runs_v2/strict/histology/executions/task-15-13-20261026T134933767825Z/task-15.13-report.json`，`strict_candidate`，provenance 不全）。
  - HyperLeaf 嵌套 LOVO 3 seed pooled OOF R²（`runs_v2/remediation/v1/deterministic-float32/hyperleaf-nested/executions/task-24-2-20261027T101633949873Z/task-24.2-report.json`，`primary_strict`）：
    `train-fold-pca` GW 0.6227 / Gsw 0.0203 / Phi 0.4626；`rgb-pca-consistency-0.05` GW 0.5921 / Gsw 0.1013 / Phi 0.4737；
    `rgb-pca-consistency`（冻结胜出）GW 0.5070 / Gsw 0.0756 / Phi 0.4670。
- 论文核心机制在 `src/hypercross_routeA.py:118` 为**单 token 退化注意力**（softmax 恒 1，Q/K 无梯度）。

### 清理（已执行）
- 删除 `runs/` 与 `runs_v2/` 下全部中间 checkpoint（`.pt`/`.pth`/`.ckpt`）共 6303 个文件、**25.60 GB**，
  以及 101 个 `__pycache__` / `.pytest_cache` / `.hypothesis` / `.tmp/pytest-*` 缓存目录。
- `runs_v2` 由 26.88 GB 降至 1.38 GB。**所有 JSON / JSONL / Markdown 证据保留**，未删除任何 report、aggregate 或 prediction artifact。
- 未删除：`runs/` 与 `runs_v2/` 的目录结构与全部证据文件；`src/`、`baselines/`、`audit/`、`strict/` 的 report。

### W1 实验（启动）
- 脚本：`experiments/wave1/hyperleaf_lovo_screen.py`（exploratory 车道）。
- 协议：leave-one-variety-out 4 折，训练集内只用 train-fold 拟合标准化/目标归一化；不读取 `solution.csv`。
- 候选：`ridge-spec` / `ridge-phys` / `mlp-spec` / `mlp-phys` / `cnn-rgb` /
  `fusion-concat` / `fusion-valueonly` / `fusion-crossattn` / `dann-spec`。
- 关键对照：`fusion-crossattn` vs `fusion-valueonly` vs `fusion-concat` 用于**因果判定**注意力是否真的有用。
- 结果目录：`runs_v3/wave1/`。

---

## 2026-11-04 — W1 结果（HyperLeaf LOVO，3 seed 筛选）

脚本 `experiments/wave1/hyperleaf_lovo_screen.py`，`--seeds 7,42,123 --epochs 200`，
结果 `runs_v3/wave1/screen-3seed_summary.{md,json}`。指标：pooled OOF R²（原始目标单位，4 折全部拼回）。

| candidate | GrainWeight | Gsw | PhiPS2 | 最差 fold 的 Gsw |
|---|---:|---:|---:|---:|
| `dann-spec`（品种对抗） | 0.6442 | **-0.0200** | 0.4366 | -0.0215 |
| `mlp-spec` | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `fusion-valueonly` | 0.5861 | -0.1425 | 0.4159 | -0.1551 |
| `fusion-crossattn` | 0.5743 | -0.1529 | 0.4223 | -0.1931 |
| `fusion-concat` | 0.5759 | -0.1628 | 0.4104 | -0.1747 |
| `ridge-spec` | 0.3439 | -0.1654 | 0.4800 | -0.1654 |
| `mlp-phys` | 0.4851 | -0.2306 | 0.5074 | -0.2787 |
| `cnn-rgb` | -0.0387 | -0.4339 | -0.0564 | -0.5369 |
| `ridge-phys` | 0.0794 | -1.2088 | 0.3143 | -1.2164 |

### 结论（三条，都会影响后续选题）

1. **注意力卖点再次被否**：`fusion-crossattn`（真 token 级双向注意力）在 GW 上
   0.5743，**低于** `fusion-valueonly`（0.5861，去掉 softmax 的纯 value 投影）和
   `fusion-concat`（0.5759，无注意力）。三者在偏差范围内同阶，注意力没有带来增益。
   → 预登记判据 A 不成立，记为 **negative evidence**；论文不应以 cross-attention 为头条机制。
2. **跨品种 Gsw 全线为负**：本 harness 家族内最好的 Gsw 是 `dann-spec` 的 -0.0200，
   仍 <0。注意这与 governed benchmark 的 `rgb-pca-consistency-0.05`（Gsw 0.1013）
   **不可直接比**：benchmark 使用 fold-fitted **PCA 伪 RGB 表征** + 一致性正则 +
   按目标路由，本 harness 没有这三样。唯一可比的是同家族内部。
   → 说明缺失的正是「PCA 视角 + 一致性」，不是模型容量。
3. **品种对抗（域泛化）是有效信号**：`dann-spec` 相对 `mlp-spec` 在 GW +0.007、
   Gsw +0.099、Phi +0.022，且在三个种子上同号。这是 W1 里唯一有方向性的正结果。
4. **形态（RGB）视角跨品种完全不可迁移**：`cnn-rgb` 三个目标全为负。
   → 支持"主线应以谱为主、形态只作辅助/门控"的定位。
5. **手搓物理特征反而有害**：`mlp-phys` < `mlp-spec`，`ridge-phys` 崩到 -1.21。
   → 物理先验需要用可学习方式注入（band attention / 谱形状损失），不能直接拼特征。

### 下一步（W2，已定）
在 harness 中补上缺失的三件套并在 Gsw 上验收：
fold-fitted PCA 伪 RGB 表征 → 双视角一致性正则 → 按目标路由；并与 `dann-spec` 组合。
预登记判据：Gsw pooled OOF R² ≥ **+0.10**（对齐当前最强 governed 结果）且 GW ≥ 0.62。

---

## 2026-11-06 — W2 / W3 / W4 结果与路线修正

### W2：双视角一致性（HyperLeaf LOVO，3 seed）
脚本 `experiments/wave2/hyperleaf_lovo_dualview.py`；结果 `runs_v3/wave2/dualview-3seed_summary.md`。

| candidate | GrainWeight | Gsw | PhiPS2 | 最差 Gsw |
|---|---:|---:|---:|---:|
| `mlp-spec` | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `dann-spec`（**无效行**） | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `dual-noconsistency` | 0.0780 | -0.3435 | 0.0146 | -0.3875 |
| `dual-consistency` | 0.0737 | -0.3427 | 0.0090 | -0.3914 |
| `dual-consistency-dann` | 0.0486 | -0.3863 | -0.0118 | -0.4137 |
| `pca-cnn` | -0.0126 | -0.4214 | -0.1186 | -0.4573 |

- **bug 更正**：W2 的 `dann-spec` 因键名 bug 未启用对抗项，与 `mlp-spec` 同值，该行作废；
  W1 的真实 DANN 值：GW 0.6442 / Gsw -0.0200 / Phi 0.4366。脚本已修复。
- fold-fitted PCA 伪 RGB 视角在本实现下**跨品种不可迁移**（单独 `pca-cnn` GW -0.0126），
  并入双视角后把 GW 从 0.6375 拖到 0.078；一致性正则几乎无影响。
- 与 governed `rgb-pca-consistency`（GW 0.507）差距大 → governed 的 PCA 伪 RGB 构造方式不同
  （`hyperleaf.pca-pseudo-rgb.fold-fitted.v1` vs 本 harness 的 4× 下采样像素 PCA）。
  **该差异必须回灌 `benchmark/` 复算后才能比较。**

### W3：Histology LOPO（3 seed）
脚本 `experiments/wave3/histology_lopo_screen.py`；结果 `runs_v3/wave3/lopo-3seed_summary.md`。
n=469 ROI / 13 患者 / 166 阳性；pooled OOF。

| candidate | Acc | BA | AUROC | ECE |
|---|---:|---:|---:|---:|
| `mlp-spec` | 0.8472 | 0.8286 | 0.8606 | 0.1267 |
| `fusion-concat` | 0.8152 | 0.8229 | **0.9190** | 0.1658 |
| `fusion-valueonly` | 0.8173 | 0.8200 | 0.9125 | 0.1557 |
| `dann-spec` | 0.8088 | 0.8094 | 0.8935 | **0.1008** |
| `cnn-rgb` | 0.7989 | 0.8071 | 0.9155 | 0.1805 |
| `fusion-crossattn` | 0.7918 | 0.7993 | 0.9053 | 0.1951 |
| *governed 参考（task 15.13）* | *0.8948* | *0.8714* | *0.8677* | — |

- 无候选在 BA 上超过 governed 参考（最好 0.8286）。
- 但融合模型 **AUROC 高、BA 低、ECE 高**（`fusion-concat` AUROC 0.9190 vs 参考 0.8677，BA 反而低 0.049）。
- **第三次复现"注意力无用"**：crossattn(0.9053) < valueonly(0.9125) < concat(0.9190)。

### W4：校准与决策规则（leave-one-patient-out）
脚本 `experiments/wave4/histology_calibration.py`；结果 `runs_v3/wave4/calib_summary.md`。

| candidate | AUROC | BA@0.5 | BA@fitted τ | BA@0.5+temp | BA@τ+temp | ECE raw | ECE temp |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp-spec` | 0.8605 | 0.8286 | 0.8405 | 0.8286 | **0.8433** | 0.1268 | 0.0568 |
| `fusion-concat` | 0.9198 | 0.8229 | 0.8125 | 0.8229 | 0.8250 | 0.1658 | **0.0354** |
| `fusion-valueonly` | 0.9125 | 0.8200 | 0.8079 | 0.8200 | 0.8023 | 0.1557 | 0.0548 |
| `dann-spec` | 0.8935 | 0.8094 | 0.8188 | 0.8094 | 0.8178 | 0.1008 | 0.0692 |
| `cnn-rgb` | 0.9150 | 0.8071 | 0.8082 | 0.8071 | 0.8080 | 0.1805 | 0.0552 |
| `fusion-crossattn` | 0.9053 | 0.7993 | 0.7971 | 0.7993 | 0.7679 | 0.1951 | 0.0731 |

- **温度标定是本次唯一稳定且大幅度的正结果**：ECE 从 0.166/0.127/0.181 降到
  0.035/0.057/0.055（降幅 3–5×），且不需要任何目标标签。
- 阈值搜索（同样 LOPO 拟合）只带来 ≤+0.015 的 BA 变化 → **"高 AUROC / 低 BA"不是阈值伪影**，
  而是两类模型犯的错不同。

## 路线修正（基于上述证据）

1. **注意力（Idea A）正式关闭**：三个 wave、六个模型、三组 seed 一致为负。
   论文标题/摘要不应包含 cross-attention 作为机制。
2. **域对抗（Idea B）保留但降权**：W1 有 +0.10 Gsw 的正向，W2/W3 未复现同量级增益。
   需要在 governed runner 上复算才有说服力。
3. **物理特征（Idea C）改为可学习注入**：直接拼手搓特征有害（`ridge-phys` Gsw -1.21），
   应改为波段注意力/谱形状约束，尚未测试。
4. **校准（Idea D）升级为主线候选贡献**：唯一稳定的正向机制，且天然契合
   AC 类"可信分析"期刊叙事。
5. **重要：harness 弱于 governed 管线**。W3 最好 BA 0.8286–0.8433 < governed 0.8714。
   → 后续**新机制必须实现在 `benchmark/` 内、跑在最强基线
   （`histology.local-route-a.task-only-control.v1`）之上**，轻量 harness 只用于快速筛选思路。

### 下一波（W5，待执行）
- 在 `benchmark/` 候选空间中新增：(a) 患者对抗谱分支；(b) 温度标定 + 风险控制的决策层；
  以 `local-route-a.task-only-control.v1` 为对照，走 LOPO 严格协议复算。
- 预登记判据：BA ≥ 0.88 且 ECE ≤ 0.05（同时改善决策与校准才算通过）。

---

## 2026-11-06（续）— W5 governed 基线校准 与 W6 对抗强度扫描

### W5：在**最强 governed 基线**上做校准/决策层
脚本 `experiments/wave5/governed_histology_calibration.py`；结果 `runs_v3/wave5/governed-calib_summary.md`。
数据源：`runs_v2/strict/histology/.../lopo/aggregates/<hash>/seed-consensus-oof.json`（469 ROI 逐样本概率）。
Platt 标定与阈值**均按 leave-one-patient-out 拟合**，不重训。

| candidate | AUROC | BA@0.5 | BA@τ(raw) | BA@τ(platt) | ECE raw | ECE platt |
|---|---:|---:|---:|---:|---:|---:|
| `alignment-nce` | 0.8936 | 0.8685 | 0.8616 | 0.8300 | 0.0748 | 0.0785 |
| `task-only-control`（最强） | 0.8621 | **0.8682** | 0.8369 | 0.7977 | **0.0636** | 0.0912 |
| `mil-concat` | 0.8845 | 0.8660 | 0.8380 | 0.8245 | 0.0981 | 0.0866 |
| `rgb-only` | 0.8770 | 0.7662 | 0.7875 | 0.7624 | 0.1043 | 0.0626 |

**结论（强负结果）**：后处理 Platt 标定 + 阈值在**每一个** governed 候选上都让 BA 变差
（最强候选 0.8682 → 0.7977），ECE 也变差（0.0636 → 0.0912）。
→ governed 基线本身已经标定良好、0.5 阈值已接近最优。**Idea D（后处理校准）不能作为"改进基线"的卖点**，
但反过来证明 governed 管线质量高（也意味着 W3/W4 里 harness 的 ECE 0.13–0.20 是 harness 自身的问题）。

### W6：谱分支对抗强度扫描（HyperLeaf LOVO）
脚本 `experiments/wave6/hyperleaf_dann_sweep.py`；结果 `runs_v3/wave6/dann-sweep_summary.md`（3 seed）
与 `lam-ext-seed42_summary.md`（λ 扩展，seed 42）。

| candidate | GrainWeight | Gsw | PhiPS2 | 最差 seed 的 Gsw |
|---|---:|---:|---:|---:|
| `dann-spec-lam1.0` | 0.6057 | **+0.0523** | 0.4308 | **+0.0308** |
| `dann-spec-lam2.0` (seed42) | 0.5694 | +0.0497 | 0.3604 | +0.0497 |
| `dann-spec-lam0.3` | 0.6442 | -0.0200 | 0.4366 | -0.0215 |
| `dann-spec-lam0.1` | 0.6428 | -0.0732 | 0.4171 | -0.0855 |
| `dann-spec-lam3.0` (seed42) | 0.5190 | +0.0044 | 0.2631 | +0.0044 |
| `mlp-spec`（无对抗） | 0.6375 | -0.1191 | 0.4144 | -0.1389 |
| `dann-fusion-*`（全融合对抗） | ≤ -0.09 | ≤ -0.04 | ≤ -0.01 | 全部崩坏 |

**结论**
1. **Gsw 对对抗强度单调**：λ 0.1 → 0.3 → 1.0 时 Gsw = -0.073 → -0.020 → **+0.052**，
   且在 λ=1.0 时**三个 seed 全部为正**（最差 +0.0308）。λ 再增大（2.0 持平、3.0 退化）会牺牲 GW/Phi。
2. **对抗必须只作用于谱分支**：把 GRL 加到融合表征上会彻底破坏 GW（-0.09 ~ -0.48）。
   这本身是一个有价值的机制性结论（多模态融合表征不能整体做域对齐）。
3. 与 governed 对比：Gsw 0.0523 优于 `train-fold-pca` 的 0.0203（+0.032），
   但 GW 0.6057 低于其 0.6227（-0.017）；尚未超过 governed 的 Gsw 专用配置 0.1013。
4. W6 预登记判据（Gsw ≥ 0.10 且 GW ≥ 0.60）**未完全通过**，记为 development 级正向证据。

## 路线修正 v2

- **唯一可复现的正向机制 = 谱分支品种对抗（λ≈1.0）**。其余（注意力、物理特征、后处理校准、全融合对抗）
  均为负结果，已记录，不再投入。
- **下一步 W7**：
  1. 把「谱分支 DANN（λ=1.0）」实现进 `benchmark/` 的 HyperLeaf 候选空间，走 governed LOVO 复算，
     取得 `strict_candidate` 以上等级；
  2. 与一致性正则/PCA 视角组合，检验能否同时保住 GW ≥ 0.62 与 Gsw ≥ 0.05；
  3. 按目标路由：GW/Phi 用干净分支、Gsw 用对抗分支（选择只用 inner varieties）；
  4. Histology 侧做患者对抗的 governed 版本，检验 BA 能否超过 0.8714。
- **投稿定位随之调整**：故事应从"跨模态融合"改为
  **"小样本科学 HSI 的域/品种不变谱表征与可迁移回归"**，卖点是
  「多模态表征不可整体域对齐、只有谱分支可对齐」+「对抗强度与 Gsw 的单调关系」。

---

## 2026-11-06（再续）— W7 路由成功 / W8 患者对抗呈反向

### W7：按目标路由 + 集成（HyperLeaf LOVO，3 seed）
脚本 `experiments/wave7/hyperleaf_routed_adversarial.py`；结果 `runs_v3/wave7/routed_summary.md`。
路由**只用训练品种做 inner LOVO 选择**，held-out 品种不参与任何选择。

| candidate | GW | Gsw | PhiPS2 | 最差 seed GW | 最差 seed Gsw |
|---|---:|---:|---:|---:|---:|
| **`routed`** | **0.6504** | **+0.0523** | 0.4149 | 0.6386 | +0.0308 |
| `dann1.0` | 0.6057 | +0.0523 | 0.4308 | 0.5907 | +0.0308 |
| `ensemble`（clean+dann0.3+dann1.0 平均） | 0.6468 | -0.0051 | 0.4394 | 0.6405 | -0.0189 |
| `dann0.3` | 0.6442 | -0.0200 | 0.4366 | 0.6383 | -0.0215 |
| `clean` | 0.6375 | -0.1191 | 0.4144 | 0.6293 | -0.1389 |

**这是本程序迄今最好的联合结果**：
- `routed` 同时达到 GW 0.6504 **且** Gsw +0.0523，三个 seed 的 Gsw 全为正，
  GW 三个 seed 全部 ≥ 0.6386。
- 对比 governed 现有最优：`train-fold-pca`（GW 0.6227 / Gsw 0.0203）、
  `gsw::rgb-pca-consistency-0.05`（GW 0.5921 / Gsw 0.1013）、冻结胜出
  `rgb-pca-consistency`（GW 0.5070 / Gsw 0.0756）。
  → `routed` 在 GW 上**超过 governed 所有单候选**，且 Gsw 远优于 GW 最优候选。
- **路由本身是可解释的**：12 个 outer 折中，**Gsw 100% 选择 `dann1.0`**；
  GW 多数选择 `clean`；PhiPS2 在 clean/dann0.3 间摇摆。
- 简单集成反而伤害 Gsw（-0.0051），因为把对 Gsw 极差的 clean 分支混了进来。

预登记判据（GW ≥ 0.62 且 Gsw ≥ 0.05，三 seed 全部满足）：GW 侧全部满足；
Gsw 均值满足但最差 seed 为 +0.0308（< 0.05）→ **部分通过**。

### W8：患者对抗（Histology LOPO，3 seed）
脚本 `experiments/wave8/histology_patient_dann.py`；结果 `runs_v3/wave8/patient-dann_summary.md`。

| candidate | Acc | BA | AUROC | ECE |
|---|---:|---:|---:|---:|
| `mlp-spec`（干净） | 0.8472 | **0.8286** | 0.8606 | 0.1267 |
| `fusion-concat`（干净） | 0.8159 | 0.8258 | **0.9183** | 0.1688 |
| `dann-spec-0.3` | 0.8088 | 0.8094 | 0.8935 | 0.1008 |
| `dann-spec-1.0` | 0.7868 | 0.7909 | 0.8745 | 0.0805 |
| `fusion-dann-spec-1.0` | 0.7754 | 0.7785 | 0.8675 | 0.1778 |
| `fusion-dann-all-1.0` | 0.5842 | 0.4590 | 0.4986 | 0.1324 |

- **患者对抗在 Histology 上是负向的**：λ 越大 BA 越低（0.8286 → 0.8094 → 0.7909）。
  与 HyperLeaf 上品种对抗的正向相反。可解释：脑胶质瘤的肿瘤/对照信号**本身与患者强相关**，
  把患者信息抹掉会同时抹掉判别信号；而小麦品种是纯 nuisance。
  副产物：对抗显著改善校准（ECE 0.127 → 0.081）。
- **再次确认"不可对融合表征做域对齐"**：`fusion-dann-all-1.0` 直接崩到 BA 0.459 / AUROC 0.499（≈随机）。
  这是继 W6 之后的第二次独立复现。

## 关键科学结论（可写进论文）

1. **跨域对齐的作用取决于"域 = nuisance 还是 = 信号"**：小麦品种是 nuisance → 对抗有益；
   病理患者与标签纠缠 → 对抗有害。两者都可复现、方向相反，构成一个干净的对照论证。
2. **域对齐必须只作用于谱分支**：对融合表征做 GRL 在两套数据上都灾难性失败。
3. **按目标路由 + 强度单调的对抗谱分支**给出迄今最优联合结果（GW 0.6504 / Gsw +0.0523）。
4. cross-attention、手搓物理特征、后处理校准均无效（各自 2–3 次独立复现）。

## 下一步（W9）
把 HyperLeaf 的 `routed`（谱分支 DANN λ=1.0 + inner-LOVO 按目标路由）实现进
`benchmark/` 候选空间，走 governed `leaf.lovo.nested.v1` 复算，取得 `strict_candidate` 以上等级；
并在 Histology 侧把"干净谱分支"作为对照固定下来（因为对抗在此为负）。
---

## 2026-11-06（审计）— 代码级复核与基线一致性问题

本轮为只读审计，未新增训练。复核结论（均已定位到具体代码行）：

1. **退化注意力确认属实**（`src/hypercross_routeA.py:122-123, 138, 145, 157`）：
   `vt=zv[:,None]; st=zs[:,None]` 使每分支只有一个 token，`softmax` 恒为 1，Q/K 梯度为 0，
   `attn_vs/attn_sv` 退化为纯 value 线性投影。同文件 `TokenCrossModalRouteA`(L184)、
   `NonlocalTokenCrossModal`(L202+)、`SpatialTokenLocalAttention`(L280) 才是真正的 token 版本；
   governed `benchmark/models/histology.py` 走 `local_pca_tokens (16,128)` 的 token 路径，
   并已内置 `HistologyAttentionDisabledControl` 作为无注意力对照。

2. **"注意力无增益"的负结果证据强度被低估**：W1/W3 的 `CrossAttnFusion` 用的是
   **真实 patch-token × band-token** 双向注意力（`experiments/wave1/hyperleaf_lovo_screen.py:201-216`），
   且配有 `fusion-valueonly`（去 softmax、等容量 value 投影对照）。
   因此"非退化注意力仍无增益"是可写进论文的强负结果，而非实现缺陷所致。

3. **跨协议比较风险（最高优先级）**：探索 harness 的**纯谱 MLP** GW=0.6375，高于 governed
   多模态候选 `rgb-pca-consistency` 的 GW=0.5070（也略高于 `train-fold-pca` 0.6227）。
   两者协议/表示/预算不同（204 波段 vs 128 波段采样、200 epoch vs inner-OOF 选 epoch），
   目前不可比较。**必须把纯谱 MLP 与融合分支放进同一 governed 管线**，否则
   "多模态融合有增益"与"谱分支对抗有增益"两个结论都无法立足。

4. **根目录旧文档数值不可与 governed 数值并列**：`CCF_A_COMPARISON.md` 的 Local Route A 0.9189 /
   官方测试 0.4550 来自 patient split、随机 split 与 legacy token 变体，与 `runs_v2` 的 LOPO
   BA 0.8714 不是同一协议；投稿前统一到 LOPO/LOVO，legacy 行仅作出处标注。

5. **机制新颖性边界**：GRL、分支级部分对齐、按目标路由（≈门控 MoE）均属已知方法。
   目前唯一具备新意的候选卖点：(a) 退化注意力的可证伪发现；(b) **域与标签是否纠缠决定对齐符号**
   （品种=nuisance→有益；患者=label-entangled→有害，两端均可复现）。(b) 需从"观察"升级为"可检验判据"。

### 由此确定的下一步（按优先级）

- **P0**：新建冻结候选空间（如 `hyperleaf.task-25.nuisance-alignment.v1`，`frozen_utc` 先于运行声明），
  把 (i) 谱分支品种对抗（λ∈{0.3,1.0}）、(ii) **纯谱 MLP 对照**、(iii) 现有 `rgb-pca-consistency`
  置于同一 `leaf.lovo.nested.v1`，3 seed 复算后再 strict，以消除第 3 条的一致性风险。
- **P1**：把"按目标路由"升级为**原理**：用 inner varieties 计算每个目标的"品种可解释方差"
  (nuisance dependence)，检验它能否**预测**哪个目标需要对冲。这是把启发式路由变成判据的关键，成本低。
- **P2**：加第三个数据集 + **合成对照**（域为 nuisance 与域与标签纠缠两种设定，扫描纠缠强度），
  把"两个数据集的轶事"升级为"可迁移规律"，是达到 ICML/NeurIPS 量级性价比最高的一步。
- **P3**：方法升级——从"只对齐谱分支"到"**只对齐 nuisance 子空间**"：用 inner domains 估计
  label-predictive 子空间（PLS/LDA 方向），GRL 仅作用于其正交补。该形式可同时解释 W6（有益）与 W8（有害）。

---

## 2026-11-06（W9）— 治理选择仪器的有效性研究 + task-24.4 复核（结论修正）

脚本 `experiments/wave9/hyperleaf_instrument_proxy_study.py`；
结果 `runs_v3/wave9/instrument-proxy_summary.{json,md}`（3 seed / 200 epoch / n=1590）。

### 先决事实：task 24.4 **已经执行过**（此前被漏读）

`runs_v2/optimization/hyperleaf/v1/deterministic-float32/latest-task-24.4-summary.json`：
`status=complete`，`all_contexts_fell_back=true`，四个 outer 折**全部回退**到冻结候选
`fallback::task-24.2::rgb-consistency-pca`，`official_evaluation_budget_consumed=0`。
`negative-evidence.jsonl`（53 KB）记录 8 个候选全部在 20-epoch 配对筛选被剪除。
各候选 inner-OOF Gsw 落在 `[-0.081, -0.079]`（极差 0.0014），最优 epoch=4/20。
→ 即 **24.3 优化空间无一提案通过 promotion guard**。

### W9 设计（同一批样本/族/seed，只改内层折构造）

| 设计 | 训练域 | 验证域 | 对应 |
|---|---|---|---|
| A `outer-lovo` | 3 个品种 | 1 个完全留出品种 | 论文报告的估计量 |
| B `inner-lovo` | 2 个品种 | 第 3 个品种 | **governed 24.4 的选择仪器** |
| C `inner-stratified` | 3 个品种（各 2/3） | 每个品种留出的 1/3 | 难度匹配的替代仪器 |

### Pooled Gsw R²（3 seed 均值）

| family | A outer | B inner-lovo | C inner-stratified |
|---|---:|---:|---:|
| `ridge-spec` | -0.1654 | -0.2405 | 0.2707 |
| `clean-mlp` | -0.1191 | -0.2195 | 0.2732 |
| `dann1.0`（谱分支品种对抗） | **+0.0523** | -0.0181 | 0.2164 |
| `clean-mlp` GrainWeight | 0.6375 | 0.5909 | 0.7760 |
| `dann1.0` GrainWeight | 0.6057 | 0.5169 | 0.6739 |

### Δ(dann1.0 − clean-mlp) on Gsw

| 设计 | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| A outer-lovo | -0.0318 | **+0.1714** | +0.0164 |
| B inner-lovo | -0.0740 | **+0.2014** | -0.0190 |
| C inner-stratified | -0.1021 | **-0.0568** | -0.0739 |

### A 的逐品种 Gsw（3 seed 均值）

| family | Heerup | Kvium | Rembrandt | Sheriff |
|---|---:|---:|---:|---:|
| `clean-mlp` | 0.0521 | 0.1122 | -0.1568 | -0.8762 |
| `dann1.0` | 0.1635 | 0.0911 | -0.0136 | **-0.3046** |

### 结论（其中一条**推翻了本轮此前的假设**，以证据为准）

1. **A 完全复现历史**：`clean-mlp` Gsw -0.1191 与 wave-1 `mlp-spec` 的 -0.1191 逐位一致；
   `dann1.0` +0.0523 与 wave-6/7 的 +0.0523 一致，且三个 seed 全为正
   （+0.0611 / +0.0651 / +0.0308）。→ 探索 harness 可复现，正向机制得到**第三次独立确认**。
2. **仪器"水平"不可迁移**：clean-mlp 在 B 上仅 -0.2195，比 A 低约 0.10。
   2 品种训练是显著更难的估计量 → 治理侧的 Gsw **绝对值**不能与报告值并列。
3. **推翻"仪器退化"假设（重要修正）**：B 上 DANN 的增益是 **+0.2014**，比 A 的 +0.1714 **更大**。
   → governed 内层仪器**并非无鉴别力**，真实机制差异它完全测得到。
4. **24.4 全体回退的真实原因**：24.3 空间里的 8 个候选（门控融合、异方差损失、
   目标均衡损失、PHI 路由、品种鲁棒归一化、consistency-0.10）都是**同一架构/表示内的变体**，
   彼此 inner-OOF 极差仅 0.0014 → guard 永远无法被跨越。**问题在候选空间"机制为空"，不在仪器。**
5. **P3 被否，且给出一个更有价值的负结果**：设计 C（难度匹配但训练域含全部品种）
   **符号反转**：Δ = -0.0568（clean 0.2732 > dann 0.2164）。因为 C 中验证样本与训练样本同品种、
   可交换，域对齐不再需要、只损失容量 —— 与 W8 Histology（域与标签纠缠→对齐有害）同型。
   → **"仅靠难度匹配不足以构造有效选择仪器"**：只有验证域真正未见时，域对齐才产生增益。
6. 副作用信号：DANN 在 GW 上稳定损失 -0.032（A）/ -0.074（B），
   → 单分支 DANN 不能直接过 promotion guard；必须**按目标路由**（Gsw 走对抗分支、GW/Phi 走干净分支）。

### 对下一步的判定（P0 因此被加强，而非推迟）

- 仪器有效 → 把「谱分支品种对抗」放进 governed 候选空间**有可被 gate 检出的增益**（预测 Δ≈+0.20 on inner Gsw）。
- 但要过 promotion guard，提交的候选必须是 **routed** 形式：
  预声明 `target_routes=(clean, adversarial, clean)`，使 Gsw 取对抗分支、GW/Phi 取干净分支，
  从而 Gsw 大幅优于守卫、GW/Phi 不退化。
- 新空间需冻结声明（`frozen_utc` 先于运行），并把**谱分支-only 对照**一并登记，
  以在同一管线内解决 GW 0.6375(探索) vs 0.5070(governed rgb-pca-consistency) 的口径冲突。

---

## 2026-11-08（W10 + task-25.1）— 统一管线确认性实验：谱域对抗的增益未在治理管线复现，且其符号由谱归一化决定（主线结论修正）

本轮完成两件事：(A) 把「谱分支品种对抗」放进 governed 管线做预登记确认性实验；(B) 用表示敏感性
网格把增益来源拆开。**结论落在证据上，并修正了 2026-11-06 节「DANN 有益已被三次独立确认」的说法。**

### A. task 25.1（预登记 · 固定预算 · 外折确认性 · development 级）

- 预登记（先于任何训练冻结）：`configs/optimization/v2/hyperleaf-task-25-preregistration.json`，
  hash `69783c2d5c392a41f1ad1ffa04963eb7f53b620e6ffa4c27b0f350826ce781e8`
- 运行器 `benchmark/runners/hyperleaf_nuisance_alignment.py`（本次新写，842 行）；
  模型 `benchmark/models/hyperleaf_nuisance_alignment.py`
- 协议：`leaf.lovo.nested.v1` 外折 4 折（hash `b22c46f6…`），seeds 7/42/123，**固定 60 epoch、
  无早停、无按外折指标选配置**；目标归一化只用 outer-train 折（governed `OUTER_TRAIN` artifact）
- 结果 `runs_v2/strict/hyperleaf-nuisance-alignment/v1/deterministic-float32/task-25-1-20261108T130712621781Z/`

| arm | 角色 | GrainWeight | Gsw | PhiPS2 |
|---|---|---:|---:|---:|
| `control::spectral-only::mse` | matched control | **0.5313 +/- 0.0012** | -0.0200 +/- 0.0166 | 0.1017 +/- 0.0350 |
| `arm::spectral-dann-lam0.3` | optimization | 0.5310 +/- 0.0131 | -0.0331 +/- 0.0008 | 0.0595 +/- 0.0027 |
| `arm::spectral-dann-lam1.0` | optimization | 0.4975 +/- 0.0073 | -0.0343 +/- 0.0054 | 0.0551 +/- 0.0111 |
| `arm::spectral-routed-dann-lam1.0` | optimization | 0.5298 +/- 0.0012 | -0.0411 +/- 0.0185 | 0.0622 +/- 0.0072 |
| `reference::rgb-pca-consistency-010` | in-run reference | 0.4991 +/- 0.0080 | **+0.0961 +/- 0.0030** | **0.4660 +/- 0.0064** |
| `reference::train-fold-pca` | in-run reference | **0.6195 +/- 0.0026** | +0.0617 +/- 0.0151 | 0.4524 +/- 0.0027 |

预登记判据：**C1 NO（-0.0143）/ C2 NO（-0.0212）/ C3 NO / C4 YES**。
C4 通过很关键：两个 in-run reference 的 Gsw（+0.0961 / +0.0617）落回 governed task-24.2 的
区间（`rgb-pca-consistency-0.05` 0.1013，冻结胜出 0.0756），

1. 证明**此运行器与 24.2 管线同口径**（首次在**同一份代码路径、同一份表示、同一预算**下同时给出
   纯谱对照与多模态参考，消解了 2026-11-06 审计第 3 条的口径冲突）；
2. 因此 C1 的失败**不能归因于 harness**，只能归因于被检验的机制本身。

**A 的最重要副产品（此前被掩盖）**：governed 管线中多模态/形态通路全面压过纯谱通路 ——
`rgb-pca-consistency` 的 Gsw 比纯谱对照高 **+0.116**（0.0961 vs -0.0200），PhiPS2 高 **+0.364**
（0.4660 vs 0.1017）；而 GrainWeight 由 `train-fold-pca` 最高（0.6195，与 24.2 的 0.6227 几乎一致）。
即：**在治理管线下，Gsw/PhiPS2 的可复现信号来自叶片表型/形态（RGB 与 train-fold-PCA 伪彩色），
不来自平均光谱的对抗对齐。**

### B. W10 表示敏感性（exploratory）— 增益的符号由谱归一化决定，与波段数无关

脚本 `experiments/wave10/hyperleaf_representation_sensitivity.py`；
结果 `runs_v3/wave10/representation-sensitivity_summary.{json,md}`。
固定模型、折、seed、预算（60 epoch），**只改谱归一化**，204 波段来自 `cache_leaf_spec_rgb.npz`。

| variant | 波段 | 归一化 | clean Gsw | dann1.0 Gsw | ΔGsw | 逐 seed ΔGsw |
|---|---:|---|---:|---:|---:|---|
| `fold-z-204` | 204 | 折内拟合逐波段 z-score（wave-1/9 表示） | -0.0476 | +0.0201 | **+0.0677** | 全为正 |
| `fold-z-128` | 128 | 折内拟合逐波段 z-score | -0.0295 | +0.0444 | **+0.0739** | 全为正 |
| `sample-snv-204` | 204 | 逐样本 SNV | +0.1154 | +0.0088 | **-0.1066** | 全为负 |
| `sample-snv-128` | 128 | 逐样本 SNV（≈ governed 表示） | **+0.1165** | +0.0529 | **-0.0636** | 全为负 |

三个直接结论：

1. **符号随归一化翻转，与波段数无关**：折内逐波段 z-score → 对抗**有益**；逐样本 SNV → 对抗**有害**。
   三种子符号完全一致（Q1/Q2/Q3 全部命中）。
2. **逐样本 SNV 本身是迄今最大的一笔单点收益**：clean-mlp 的 Gsw 从 -0.048/-0.030 抬到
   +0.115/+0.117，**ΔR² ≈ +0.16**，远超任何对抗对齐带来的增益（+0.07）。
3. **预算只影响幅度、不改变符号**：同一 `fold-z-204` 表示下，60 epoch Δ=+0.0677、200 epoch Δ=+0.1714，
   两者三 seed 同号（新增 `runs_v3/wave9/instrument-proxy-e60_summary.md` 作为 60-epoch 对照）。

机理自洽的解释（作为可检验假设记录，不是结论）：逐波段 z-score 保留了**跨样本的幅度/水平**，
网络因此可以靠"记住品种水平"降低训练误差，而这正是跨品种 Gsw 的失败模式 → 对抗对齐抵消它；
逐样本 SNV 已把水平除掉，表示里剩下的品种信息与目标**纠缠**，对抗对齐只会删掉有用信号。
这与 W8（病理患者域与标签纠缠 → 对齐有害）是**同一机制的两端**，且现在**在同一个数据集内、
只用一个归一化开关就能复现两端** —— 比此前"两个数据集的两段轶事"强得多。

### C. 对主线结论的修正（覆盖 2026-11-06 节的相关表述）

- 2026-11-06 写道「谱分支品种对抗（GRL λ≈1.0）是唯一可复现的正向机制」。**修正为**：
  该增益是**表示条件性的**（仅在折内逐波段 z-score 空间成立），在治理管线的逐样本 SNV 表示下
  **不复现且为负**。三次"复现"共享同一个表示，因此那次确认是**同一条件下的重复**，不等于跨条件稳健。
- 真正稳健且可复现的正向机制是**表示/归一化设计**（逐样本 SNV + 形态通路），不是对抗对齐。
- 「域对齐是否有益取决于域与标签是否纠缠」这一命题**被本轮加强**：现在有 4 组证据
  （小麦-折内 z / 小麦-SNV / 病理患者 / wave-9 分层设计），且小麦侧只改归一化即可双向复现。

### D. 待办与下一步（本轮判定）

- **P0-a（在进行中）**：governed 管线 200-epoch 诊断（`control::spectral-only::mse` + `dann-lam1.0`），
  用于分离"治理配置/预算"与"表示"两个混淆。注意：governed 的纯谱对照 Gsw -0.0200 比 B 中
  **同一表示**的 `sample-snv-128` clean-mlp（+0.1165）低 **0.14 R²**，而同一运行里的 RGB/PCA 参考
  能达到 +0.0961 —— 说明这不是全局预算问题，而是**纯谱分支的配置问题**（lr 2e-4 / bs 64 /
  64 维 `_SpectrumEncoder` / mild 风格增强 vs 探索侧 lr 1e-3 / bs 128 / 256 维 MLP）。
  该差异必须先解释清楚，否则任何"谱表示"结论都不成立。
- **P0-b**：把"逐样本 SNV"这一发现**回灌 governed 管线**并预登记复算（它是当前最便宜的大额收益）。
- **P1**：把"域-标签纠缠"从观察升级为**判据**：在 inner varieties 上估计每个目标的
  "域可解释方差"（domain-explainable variance），检验它能否预测哪个目标需要对齐；
  并实现"只对齐 nuisance 子空间"（GRL 仅作用于 label-predictive 子空间的正交补）。
- **P2**：合成对照（域为 nuisance / 域与标签纠缠两种设定，扫描纠缠强度），把规律做强。

### E. P0-a 诊断完成（200 epoch，governed 同表示）—— 60 epoch 是欠训练；对抗在 SNV 表示下稳定有害

命令 `--execute --smoke --epochs 200 --arms control::spectral-only::mse,arm::spectral-dann-lam1.0`；
输出 `runs_v2/strict/hyperleaf-nuisance-alignment/v1/smoke/task-25-1-20261108T141414395253Z/task-25.1-summary.md`

| arm（200 epoch） | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `control::spectral-only::mse` | 0.5230 +/- 0.0111 | **+0.1016 +/- 0.0049** | **+0.4781 +/- 0.0042** |
| `arm::spectral-dann-lam1.0` | 0.3474 +/- 0.0953 | -0.0386 +/- 0.0053 | 0.0535 +/- 0.0202 |

1. **60 epoch 属于欠训练**：同一 arm、同一表示，Gsw 从 -0.0200（60 ep）→ **+0.1016**（200 ep），
   PhiPS2 从 0.1017 → 0.4781。D 节记录的"governed 纯谱比探索侧低 0.14 R²"因此**主要是预算/配置问题**，
   不是表示或代码路径问题。
2. **DANN 的负号在 200 epoch 依旧成立**：ΔGsw = **-0.1402**（三 seed 全负），与 W10 在
   `sample-snv-128` 上预测的 -0.0636 同号；同轮 GW 0.3474、PhiPS2 0.0535 同时崩塌。
   → 「逐样本 SNV 表示下对抗对齐有害」现已在**两个 code path × 两个预算**下稳定复现。
3. **本轮最重要的新事实**：单分支、仅用 128 波段平均光谱、仅做逐样本 SNV 的 MLP，200 epoch 下
   Gsw **0.1016** / PhiPS2 **0.4781**：
   - Gsw 追平 24.2 最强候选 `rgb-pca-consistency-0.05`（0.1013），高于冻结胜出候选（0.0756）；
   - PhiPS2 高于**全部** 24.2 候选（最高 0.4737）；
   - GrainWeight 0.5230 低于 `train-fold-pca`（0.6227）。
4. 由于 task 24.2 的候选集合（5 个）**全部含 RGB 或 PCA 输入、没有纯谱候选**，
   "多模态/一致性带来 Gsw、PhiPS2 增益"这一叙事**尚未被证明**：一个训练充分的纯谱基线即可达到同等水平。
   24.4 全体回退也因此有了新解释线索：内层仪器 `MAX_SELECTION_EPOCH=80`、best-epoch≈4 的 epoch 策略
   与"候选空间无纯谱机制"叠加，使这条更简单的路线从一开始就不在可选集合内。
5. 该结论仍是 **development 级**（3 seed、单一表示、60/200 两个预算点），
   需要按 24.x 规范做一次预登记的同预算复算才能升级。

### F. 本轮净结论（供后续引用）

- 谱域品种对抗**不是**稳健机制：其符号由谱归一化决定（折内逐波段 z-score → 有益；逐样本 SNV → 有害），
  与波段数无关（204 / 128 同号），在治理管线的两个预算点上均为负。
- 迄今**最大且最稳健**的单点收益是**逐样本 SNV 归一化 + 足够训练预算**（Gsw ΔR² ≈ +0.16 相对折内 z-score），
  以及形态（RGB / train-fold-PCA）通路（Gsw +0.116、PhiPS2 +0.364 相对同预算纯谱对照）。
- "域对齐是否有益取决于域与标签是否纠缠"得到**同数据集内的双向复现**（只改归一化），
  这是目前唯一具备顶会级"可检验规律"潜力的命题。
- 「跨模态注意力无效」「手搓物理特征无效」「后处理校准无效」「全融合对抗有害」四条负结果不变。

---

## 2026-11-08（审计续）— 代码级证据：task 24.2/24.3 为何系统性漏掉"纯谱基线"

只读复核 `benchmark/runners/hyperleaf_optimization_registry.py` 与
`benchmark/runners/hyperleaf_inner_optimization.py`，结论如下（均定位到代码）：

1. **24.3 候选空间（9 条 = 3 matched control + 1 fallback + 5 optimization）内没有纯谱候选。**
   所有条目的 `representation_key` 都是 `RGB_SPECTRUM` 或 `PCA_SPECTRUM`；`changed_axis` 依次为
   `consistency_weight / gsw_route / phips2_route / spectral_normalization / loss_contract / loss`
   —— **representation 从未被当作优化轴**（`_build_candidates`，registry L377-430）。
2. **24.2 的 5 个候选**（`REQUIRED_CANDIDATE_IDS`：`branch::rgb-pca-consistency`、
   `gsw::rgb-pca-consistency-0.05::mse::standardized`、`branch::train-fold-pca`、
   `branch::style-augmentation`、`branch::independent-target`）**同样全部含 RGB 或 PCA 输入**。
   → 治理栈里**从未存在** matched 纯谱基线，因此"多模态/一致性带来 Gsw、PhiPS2 增益"**从未被检验**
   （不是被证伪，而是缺对照）。
3. **epoch 策略对晚收敛模型有系统性偏置**：24.3 的 `screening_schedule` 是
   `paired-screen-20-epochs`（`SCREEN_CAP=20`）→ `promotion-screen-80-epochs`（`PROMOTION_CAP=80`）。
   而本轮实测：同一纯谱 arm 在 60 → 200 epoch 之间 Gsw 从 **-0.0200 升到 +0.1016**。
4. **promotion guard 与之叠加**（`PromotionGuard`）：要求 Gsw pooled-OOF 均值 Δ≥ **+0.0100**、
   **三 seed 全部严格为正**、且 GrainWeight/PhiPS2 均值不下降超过 0.0100。
   一个 200 epoch 才成形的模型在 20 epoch 屏幕上必然被剪除，且 guard 只在 inner-OOF 上生效，
   不允许 outer 复活 —— 结构性上无法让简单但晚收敛的候选胜出。
5. 这为 24.4 的观测（8 个候选 inner-OOF Gsw ∈ [-0.081, -0.079]、极差仅 0.0014、best epoch=4/20）
   提供了机制解释：空间内候选共享同一多模态家族，20 epoch 不足以区分。
6. 顺带记录一处口径细节（尚不影响结论，但需在 paper 中统一）：24.3 声明的谱归一化是
   `per-band-zscore-inner-train.v1`（保幅度），而 `hyperleaf_nested_remediation._load_nested_data`
   与 `hyperleaf_inner_optimization._load_context_data` 实际构造的 `standard_spectrum`
   是**逐样本 SNV**（`(x-mean)/std`，L397-400）——即 candidate 声明与 runner 实际输入不一致，
   需要在下一次 superseding 空间里显式对齐。

### 由此确定的 superseding 要求（写入下一步）

- 新空间必须包含 **matched 纯谱候选**（同预算、同折、同 seed），以及**表示/归一化作为一等优化轴**；
- 屏幕预算必须覆盖收敛所需（本轮证据指向 ≥200 epoch），或至少在 20/80/200 三点上验证排名稳定性；
- 候选声明中的 `spectral_normalization_key` 必须与 runner 实际输入一致。

---

## 2026-11-09（W11）— 谱表示网格：逐样本 SNV 是迄今最大单点收益；对抗增益 = "可移除 nuisance 的回填"

脚本 `experiments/wave11/hyperleaf_spectral_representation_grid.py`；
结果 `runs_v3/wave11/spectral-representation-grid_summary.{json,md}`。
外折 LOVO、204 波段缓存平均光谱、200 epoch、seeds 7/42/123、clean MLP 与 DANN(λ=1.0) 同架构同预算。

| representation | 特征数 | 域可识别率 | 线性读出 Gsw | clean GW | clean Gsw | clean PhiPS2 | dann Gsw |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fold-z-raw`（折内逐波段 z） | 204 | 0.8427 | -0.1655 | 0.6375 | -0.1191 | 0.4144 | 0.0523 |
| **`snv-raw`（逐样本 SNV）** | 204 | 0.7326 | -0.0166 | 0.6252 | **+0.1075** | **0.4800** | 0.1137 |
| `snv-z`（SNV 后再逐特征 z） | 204 | 0.8325 | -0.2917 | 0.6209 | -0.0531 | 0.4657 | 0.0286 |
| `snv-d1-z`（+一阶导） | 408 | 0.8099 | -0.8739 | 0.5714 | -0.1644 | 0.4875 | -0.0799 |
| `snv-cr-z`（+连续统去除） | 408 | 0.7919 | -0.6812 | 0.4982 | -0.1651 | 0.4535 | -0.0216 |
| `snv-d1-cr-z` | 612 | 0.8030 | -0.4055 | 0.5027 | -0.1858 | **0.5020** | -0.0769 |
| `msc-z`（见下方缺陷说明） | 204 | 0.8211 | -0.3779 | 0.6055 | -0.0821 | 0.4550 | -0.0040 |
| `snv-msc-z`（同上，重复） | 204 | 0.8211 | -0.3779 | 0.6055 | -0.0821 | 0.4550 | -0.0040 |

`fold-z-raw` 三个 seed 的 clean Gsw（-0.1061/-0.1124/-0.1389）**逐位复现** wave-1 的 -0.1191，
说明网格与历史可比。

### 结论

1. **逐样本 SNV（且不再叠加逐特征标准化）是本轮最优谱表示**：Gsw **+0.1075**、PhiPS2 **0.4800**、
   GrainWeight 0.6252；相对 `fold-z-raw` 为 **Gsw +0.2266 / Phi +0.0656 / GW -0.0123**，
   且三 seed 极稳（Gsw 0.1078/0.1057/0.1091）。**这是本项目迄今最大的单点算法收益。**
2. **对照 governed 参考框架**：`snv-raw` 的 GrainWeight 0.6252 约等于 24.2 最好的
   `train-fold-pca`（0.6227），Gsw 0.1075 高于 24.2 **全部**候选（最高 0.1013），
   PhiPS2 0.4800 高于 24.2 **全部**候选（最高 0.4737）。
   → **一个纯谱单分支模型在两个目标上超过、在第三个上打平整个 governed 多模态候选族。**
3. **导数 / 连续统去除 / SNV 后再做逐特征 z 都会变差**（线性读出 Gsw 从 -0.017 掉到 -0.29 ~ -0.87；
   MLP 同样变差）。→ 跨品种 Gsw 的信息在**平滑的 SNV 光谱形状**里，微分与连续统去除放大了 nuisance。
4. **对抗增益的本质是"可移除 nuisance 的回填"**：Δ(DANN-clean) 与 clean 模型的差成正比 ——
   `fold-z-raw` clean -0.119 → Δ **+0.171**；`snv-cr-z` clean -0.165 → Δ **+0.144**；
   `snv-d1-z` clean -0.164 → Δ +0.085；而 clean 已经很好的 `snv-raw`（+0.108）→ Δ **+0.006**。
   即：**表示里只要还留着可移除的水平/幅度 nuisance，对齐就显得有用；一旦表示把它去掉，增益就消失。**
   → 正确的做法是**改表示**，不是加对抗目标。这也解释了 24.x 为何"发现"了对抗（它的谱表示是保幅度的）。
5. 预登记判据：**Q1 部分成立**（Gsw 增益 +0.2266 远超 +0.03，但 GW 小幅 -0.0123）；
   **Q2 不成立**（域可识别率高并不能预测 Δ：`snv-cr-z` 可识别率 0.792 却有 Δ=+0.144，
   而 `snv-raw` 可识别率最低 0.733 却 Δ≈0）。第 4 条给出的"clean 弱 ⇔ Δ 大"才是更准的判据。
6. **已知缺陷**：wave-11 的变换分派对 `msc-z` 与 `snv-msc-z` 走了同一分支，
   导致两行完全相同（MSC 实际作用在已 SNV 化的光谱上）。已在 wave-12 修正为"对原始光谱做 MSC"。

---

## 2026-11-09（W12）— 修正 MSC / 新增化学计量潜表示 / 诚实的逐目标表示路由（负结果）

脚本 `experiments/wave12/hyperleaf_representation_routing.py`；
结果 `runs_v3/wave12/representation-routing_summary.{json,md}`（另 `snv-budget400_summary.md`）。
外折 LOVO、200 epoch、seeds 7/42/123、clean MLP；内层仪器为"训练 2 品种 → 验证第 3 品种"。

| representation | GrainWeight | Gsw | PhiPS2 |
|---|---:|---:|---:|
| `fold-z`（折内逐波段 z） | 0.6375 +/- 0.0058 | -0.1191 +/- 0.0142 | 0.4144 +/- 0.0085 |
| **`snv`（逐样本 SNV，200 ep）** | **0.6252 +/- 0.0009** | **+0.1075 +/- 0.0014** | **0.4800 +/- 0.0001** |
| `snv`（同表示，400 ep） | **0.6415 +/- 0.0007** | 0.0958 +/- 0.0016 | 0.4759 +/- 0.0005 |
| `snv-pca32`（SNV + 折内 32 维 PCA） | 0.2557 +/- 0.0839 | -0.3088 +/- 0.0372 | 0.4284 +/- 0.0292 |
| `msc`（对**原始**光谱做 MSC，已修正） | 0.6034 +/- 0.0028 | -0.0843 +/- 0.0186 | 0.4539 +/- 0.0010 |
| `snv-d1-cr-z`（SNV+一阶导+连续统去除） | 0.5027 +/- 0.0277 | -0.1858 +/- 0.0298 | 0.5020 +/- 0.0052 |

### 结论

1. **修正后的 MSC 与 PCA 潜表示都被 `snv` 支配**（MSC：Gsw -0.0843；PCA32：Gsw -0.3088、GW 崩塌到 0.2557）。
   → 结论不是"某种预处理技巧"，而是**逐样本 SNV 这一件事**：它同时去掉幅度 nuisance 且不破坏光谱形状。
2. **诚实的逐目标表示路由不成立（负结果）**：用内层品种证据为每个目标选表示，外层 pooled OOF 为
   GW 0.6046 / Gsw 0.1075 / PhiPS2 0.4442，**全面劣于"一律用 `snv`"**（0.6252 / 0.1075 / 0.4800）；
   而外层 oracle 路由上界是 0.6375 / 0.1075 / 0.5020。
   → 有约 +0.012（GW）/ +0.022（Phi）的 headroom，但内层仪器**抓不到**（Gsw 4/4 折都选 `snv` 说明仪器在
   Gsw 上是对的，但在 GW/Phi 上噪声导致误选）。这与 task-24.4"内层仪器无法区分候选"同源。
   → **方法层面应固定一个原则性的单一表示，而不是做表示路由**；这也否证了把 wave-7 的"按目标路由"
   从"分支轴"推广到"表示轴"的想法。
3. **`snv` 的三 seed 方差极小**（200 ep：Gsw ±0.0014、PhiPS2 ±0.0001、GW ±0.0009），是当前最稳的候选。
4. **预算存在 GW↔Gsw 的单调权衡，且已接近饱和**：200 → 400 epoch 时 GW 0.6252 → **0.6415**、
   Gsw 0.1075 → 0.0958、PhiPS2 0.4800 → 0.4759。400 epoch 的 `snv` 在 GrainWeight 上甚至高于
   `fold-z`（0.6375）。→ 不存在"再训 10 倍还会更好"的空间；但**两个预算点各自占优不同目标**，
   因此"预算快照集成"（pre-register 地在 200/400 epoch 两点平均预测）是一个便宜、无需外层选择的
   待验证提分点（各目标上界为 GW 0.6415 / Gsw 0.1075 / Phi 0.4800）。

### 对本项目主线的最终判定（用于 paper 定位）

- **稳健、可复现、量级最大的机制是"谱表示设计"**：逐样本 SNV 把跨品种 Gsw 从 -0.1191 抬到 +0.1075
  （+0.2266），PhiPS2 从 0.4144 抬到 0.4800。
- 一个**纯谱单分支**模型（`snv` + 普通 MLP + 200 epoch）即达到 GW 0.6252 / Gsw 0.1075 / PhiPS2 0.4800，
  在 Gsw、PhiPS2 上高于 governed task-24.2 的**全部**候选，在 GrainWeight 上打平最好的
  `train-fold-pca`（0.6227）；400 epoch 版本 GW 0.6415 则**超过**了 24.2 的全部候选。
- 因此当前论文故事应是**"小样本科学 HSI 的表示设计 + 诚实的跨品种迁移评估"**，
  而不是"跨模态融合/一致性"或"域对抗对齐"；后两者在本项目中分别是**从未被检验**和**表示缺陷的症状**。

### 交叉验证：`snv` 结论在 governed 与探索两条独立代码路径上一致（重要）

| 路径 | 表示 | 波段 | 编码器/优化 | Gsw | PhiPS2 | GrainWeight |
|---|---|---:|---|---:|---:|---:|
| governed task-25.1 `control::spectral-only`（200 ep） | 逐样本 SNV | 128 | `_SpectrumEncoder(64)`，AdamW lr 2e-4，bs 64 | **+0.1016** | **0.4781** | 0.5230 |
| 探索 wave-12 `snv`（200 ep） | 逐样本 SNV | 204 | MLP(256)，AdamW lr 1e-3，bs 128 | **+0.1075** | **0.4800** | 0.6252 |

两条路径**在 Gsw 与 PhiPS2 上高度一致**（差值 0.006 / 0.002），且与各自的历史基线一致；
说明"逐样本 SNV 把跨品种 Gsw 从负值抬到约 +0.10、PhiPS2 抬到约 0.48"是**跨代码路径、跨波段数、
跨编码器、跨学习率稳健的结论**。唯一显著分歧在 GrainWeight（0.523 vs 0.625），
指向**编码器/优化配置**而非表示本身 —— 这是 task 25.2 需要顺带解决的最后一个口径问题。
