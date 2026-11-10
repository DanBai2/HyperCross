# Requirements Document

## Introduction

本需求文档从已批准的 `cvpr2027-hyperspectral-benchmark` 技术设计推导 HyperCross 统一 benchmark 的可验证需求。Benchmark 的目标是把现有分散实验迁移为协议优先、prediction-first、证据可追溯且可恢复的评测系统，同时保留历史结果及其限制。

本功能覆盖三个相互独立的任务轨道：HyperLeaf2024 三目标回归、HistologyHSI-GB 患者级二分类和 Remote_Sensing_Sanity_Track。正式比较必须绑定相同的数据版本、Protocol_Manifest、输入表示、选择规则、Seed_Set 和指标定义。当前规范阶段只定义需求，不修改训练实现、不启动训练，也不生成新的性能结果。

## Glossary

- **Benchmark_System**：统一规划、执行、审计和报告 HyperCross benchmark 的系统。
- **Dataset_Catalog**：登记数据集语义、样本、分组、标签、输入位置、许可和数据指纹的组件。
- **Sample_Manifest**：按 Canonical_Order 保存 Sample_ID、Group_ID、标签引用、输入引用和内容哈希的不可变清单。
- **Sample_ID**：在一个数据集版本内唯一标识一个样本的稳定标识符。
- **Canonical_Order**：按照规范化 Sample_ID 字节序生成、不受输入枚举顺序影响的确定性样本排序。
- **Canonical_Serialization**：使用固定字段顺序、固定集合排序和固定数值编码产生的确定性序列化形式。
- **Dataset_Hash**：由数据版本和 Sample_Manifest 的 Canonical_Serialization 计算得到的内容哈希。
- **Group_ID**：用于隔离同一品种或同一患者样本的分组标识。
- **Protocol_Planner**：在模型构建前产生并校验评测划分的组件。
- **Protocol_Manifest**：保存 Protocol_Key、Outer_Fold、Inner_Fold、样本 ID、Group_ID、选择规则和评估预算的不可变协议清单。
- **Protocol_Key**：唯一标识评测协议及版本的稳定名称。
- **Protocol_Hash**：由 Protocol_Manifest 的 Canonical_Serialization 计算得到的内容哈希。
- **Fold_ID**：在一个 Protocol_Manifest 内唯一标识 fold 的稳定名称。
- **Outer_Fold**：只在模型与选择规则冻结后用于一次外层评估的数据划分。
- **Outer_Train_Set**：Outer_Fold 中允许训练、拟合和内部选择使用的样本集合。
- **Outer_Test_Set**：Outer_Fold 中仅允许最终评估使用的样本集合。
- **Inner_Fold**：完全位于 Outer_Train_Set 内、用于候选配置和 epoch 选择的数据划分。
- **Official_Test_Set**：HyperLeaf2024 官方协议中冻结的 820 个测试样本集合。
- **Evaluation_Budget**：Protocol_Manifest 声明的最终测试评估次数上限。
- **Selection_Decision**：基于 Inner_Fold 证据冻结的 epoch、模型、路由、blend、阈值或 calibration 选择记录。
- **Fitted_Artifact**：从指定训练样本估计得到并记录 Fit_Sample_IDs 的预处理、阈值或 calibration 产物。
- **Fit_Sample_IDs**：参与估计 Fitted_Artifact 状态的精确 Sample_ID 集合。
- **Preprocessing_Manager**：拟合、应用和审计 Fold_Scoped_Preprocessing 的组件。
- **Fold_Scoped_Preprocessing**：每个 fold 独立拟合的目标归一化、PCA、波段归一化、插补、类别权重、阈值和 calibration。
- **Feature_Cache**：保存确定性特征或 Fitted_Artifact 输出及 Lineage 的缓存。
- **Lineage**：记录输入哈希、配置哈希、Fit_Sample_IDs、变换哈希和输出 shape 的来源链。
- **Model_Registry**：使用稳定 Model_Key 解析模型实现、输入契约和 Provenance 的组件。
- **Model_Key**：唯一标识模型实现和版本的稳定 registry key。
- **Model_Identity**：包含 Model_Key、显示名、类全限定名、实现类别、参数量和来源 revision 的模型身份记录。
- **Input_Contract**：声明模型必需模态、Representation_Key、每个模态的 tensor shape、dtype、通道或波段顺序和预处理 key 的契约。
- **Representation_Key**：唯一标识 RGB、全局光谱、局部光谱、PCA pseudo-RGB 或 Full_Cube 等输入表示的名称。
- **Provenance**：代码 revision、上游仓库、上游 commit、许可、源码哈希、补丁哈希、环境哈希、配置哈希、数据哈希和 artifact 哈希等来源信息。
- **Classic_Model**：PLSR、RF、SVM-RBF 等经典统计或机器学习基线。
- **In_Repo_Model**：由当前仓库维护的 HyperCross 或匹配控制模型。
- **Official_Model**：固定上游仓库和 commit，并通过上游复现审计的模型实现。
- **Adapted_Official_Model**：在保留上游身份与补丁记录的情况下适配到本项目输入的官方模型。
- **Proxy_Model**：结构受现代模型启发但不是上游官方实现的紧凑代理模型。
- **Matched_Input_Control**：与候选模型使用相同输入表示和可比训练预算的控制模型。
- **Training_Engine**：使用冻结配置进行训练、内部选择、重拟合和 checkpoint 管理的组件。
- **Prediction_Artifact**：逐样本保存 Sample_ID、Fold_ID、真实值、预测值、概率和 Inference_Branch 的不可变预测文件。
- **Prediction_Key**：由 Sample_ID、Fold_ID、seed 和 Inference_Branch 组成的 Prediction_Artifact 唯一键。
- **Inference_Branch**：正式推理时固定使用的模型输出分支。
- **Metric_Engine**：仅从通过完整性校验的 Prediction_Artifact 计算、聚合和比较指标的组件。
- **Run_Record**：保存单次运行身份、协议、配置、artifact、指标、状态和 Claim_Status 的记录。
- **Run_ID**：唯一标识一个运行记录的不可复用标识符。
- **Run_Registry**：以 append-only 方式保存 Run_Record、导入历史证据和登记 supersession 的组件。
- **Aggregate_Record**：由明确 Member_Run_IDs 的 Prediction_Artifact 聚合而成的结果记录。
- **Publication_Aggregate**：通过完整 fold、完整 Seed_Set 和 Claim_Gate 校验后允许进入发布输出的 Aggregate_Record。
- **Member_Run_IDs**：构成 Aggregate_Record 的完整 Run_ID 集合。
- **OOF_Prediction**：样本只在未参与对应模型拟合的 fold 中产生的 out-of-fold prediction。
- **Seed_Set**：协议预先声明并用于可发布聚合的随机种子集合。
- **Report_Generator**：从 Aggregate_Record 生成 JSON、CSV、Markdown 和审计报告的组件。
- **Claim_Gate**：依据协议、Provenance、完整性和控制实验限制对外结论级别的组件。
- **Claim_Status**：`primary_strict`、`strict_candidate`、`development`、`exploratory`、`sanity_only`、`proxy_only` 或 `invalid_or_leaky` 中的一个证据状态。
- **Claim_Status_Ceiling**：证据在某项限制下可获得的最高 Claim_Status；同时适用多个 ceiling 时使用允许级别最低的 ceiling。
- **Legacy_Evidence**：由现有结果文件、历史 runner 或 checkpoint 导入且可能缺少完整 Provenance 的证据。
- **Superseding_Record**：在不改写旧记录的情况下解释并替代旧结论的新 Run_Record 或 Aggregate_Record。
- **Recovery_Manager**：支持原子写入、失败登记、断点恢复和缺失作业重跑的组件。
- **Canonical_Completed_Record**：给定 Fold_ID、Model_Key、seed 和配置哈希后唯一用于聚合的完成记录。
- **Frozen_Run_Identity**：由 Dataset_Hash、Protocol_Hash、Model_Key、配置哈希、Fold_ID、seed 和 Environment_Lock 兼容状态组成的不可变运行身份。
- **Resource_Budget**：对训练步数、epoch、运行时长、设备内存、参数量或输入体积设置的有限正整数限制。
- **Hard_Limit**：超出后必须在 checkpoint 边界停止运行的 Resource_Budget 限制。
- **Advisory_Limit**：超出后记录告警但不要求停止运行的 Resource_Budget 限制。
- **Reproduction_Gate**：Official_Model 进入项目适配评测前必须通过的上游结果、版本、许可和环境检查。
- **CVPR_Claim**：拟写入 CVPR 2027 论文主表、摘要或 headline conclusion 的性能或方法结论。
- **HyperLeaf_Official_Protocol**：使用 1,590 个标注训练样本进行内部按品种选择、全量重拟合并对 820 个 Official_Test_Set 样本评估的协议。
- **HyperLeaf_LOVO_Protocol**：以 wheat variety 为 Group_ID 的严格 nested leave-one-variety-out 协议。
- **Histology_Nested_Patient_Protocol**：以 patient 为 Group_ID 的外层患者划分和内层患者选择协议。
- **Histology_Fixed_Epoch_LOPO_Protocol**：预先冻结 epoch 后逐患者留出的 leave-one-patient-out 协议。
- **Remote_Sensing_Sanity_Track**：只验证上游实现和移植流程可运行的遥感数据评测轨道。
- **Testing_Suite**：执行 unit、property-based、integration、regression、smoke 和 audit 检查的测试集合。
- **Environment_Lock**：保存 Python、框架、依赖、CUDA 和设备相关版本的可重建环境记录。
- **Error_Record**：保存错误代码、失败阶段、受影响标识、期望值、实际值和不变性检查结果的机器可读记录。
- **Failure_Atomicity**：失败操作不修改失败前的 registry 状态、既有 manifest、既有 artifact 或已完成运行记录。
- **Display_Precision**：JSON、CSV 和 Markdown 发布表格中统一使用的小数位数；本规范固定为四位小数。
- **Numeric_Tolerance**：Metric_Engine 数值一致性比较使用的绝对误差上限，由 metric schema 显式声明。

## Requirements

### Requirement 1: Audit Current Evidence and Project State

**User Story:** As a benchmark maintainer, I want the existing repository evidence audited without rewriting source artifacts, so that future claims preserve the context and limitations of prior experiments.

#### Acceptance Criteria

1. WHEN Legacy_Evidence is imported, THE Run_Registry SHALL preserve the source path in the imported Run_Record.
2. WHEN Legacy_Evidence is imported, THE Run_Registry SHALL preserve the source content hash in the imported Run_Record.
3. WHEN Legacy_Evidence is imported, THE Run_Registry SHALL preserve the recorded protocol description without semantic rewriting.
4. WHEN Legacy_Evidence is imported, THE Run_Registry SHALL preserve every recorded metric value without numeric rewriting.
5. WHEN Legacy_Evidence is imported, THE Run_Registry SHALL preserve every known caveat.
6. WHEN a Legacy_Evidence source lacks a runner, THE Claim_Gate SHALL record `runner` in the missing-provenance fields.
7. WHEN a Legacy_Evidence source lacks a Sample_Manifest, THE Claim_Gate SHALL record `sample_manifest` in the missing-provenance fields.
8. WHEN a Legacy_Evidence source lacks a Protocol_Manifest, THE Claim_Gate SHALL record `protocol_manifest` in the missing-provenance fields.
9. WHEN a Legacy_Evidence source lacks a Prediction_Artifact, THE Claim_Gate SHALL record `prediction_artifact` in the missing-provenance fields.
10. WHEN a Legacy_Evidence source lacks a code revision, THE Claim_Gate SHALL record `code_revision` in the missing-provenance fields.
11. WHEN a Legacy_Evidence source lacks an Environment_Lock, THE Claim_Gate SHALL record `environment_lock` in the missing-provenance fields.
12. IF a historical result used an Outer_Test_Set for checkpoint selection, THEN THE Claim_Gate SHALL assign the `invalid_or_leaky` Claim_Status.
13. IF a historical result used an Outer_Test_Set for early stopping, THEN THE Claim_Gate SHALL assign the `development` Claim_Status ceiling unless another applicable rule imposes a lower ceiling.
14. WHEN the historical Histology single-loop patient CV result with fold-macro accuracy `0.9153 ± 0.0686` is imported, THE Claim_Gate SHALL assign the `development` Claim_Status ceiling because the outer fold participated in early stopping.
15. WHEN the historical fixed 9/4 patient split is imported, THE Claim_Gate SHALL assign the `exploratory` Claim_Status ceiling.
16. WHEN the historical `0.9392` ensemble result is imported, THE Claim_Gate SHALL assign the `exploratory` Claim_Status ceiling.
17. WHEN the existing `HybridSN_lite` result is imported, THE Model_Registry SHALL identify the associated implementation as a Proxy_Model.
18. WHEN the existing `S2Mamba_lite` result is imported, THE Model_Registry SHALL identify the associated implementation as a Proxy_Model.
19. WHEN the existing `MamTrans_lite` result is imported, THE Model_Registry SHALL identify the associated implementation as a Proxy_Model.
20. WHEN the Indian Pines S²Mamba result with OA `0.9749`, AA `0.9871`, and Kappa `0.9712` is imported, THE Claim_Gate SHALL assign the `sanity_only` Claim_Status ceiling.
21. WHEN the HyperLeaf target-routed RGB/PCA consistency result is imported before unified-runner reproduction, THE Claim_Gate SHALL assign the `strict_candidate` Claim_Status ceiling.
22. WHEN the Histology PCA-only inference result is imported before unified-runner reproduction, THE Claim_Gate SHALL assign the `strict_candidate` Claim_Status ceiling.
23. WHEN a historical selector maps a requested Model_Key to a different class, THE Claim_Gate SHALL assign the `invalid_or_leaky` Claim_Status.
24. WHEN a historical selector maps a requested Model_Key to a different class, THE Run_Registry SHALL preserve the requested Model_Key, resolved class, and mismatch explanation.
25. WHEN a Legacy_Evidence conclusion is corrected, THE Run_Registry SHALL append one Superseding_Record that references the original record identifier.
26. WHEN a Superseding_Record is appended, THE Run_Registry SHALL leave the original Run_Record and source artifact unchanged.
27. IF a Legacy_Evidence import fails validation, THEN THE Run_Registry SHALL block the append before assigning a Run_ID.
28. IF a Legacy_Evidence import fails validation, THEN THE Run_Registry SHALL emit an Error_Record containing the source path, source content hash, failing field, expected constraint, and observed value.
29. IF a Legacy_Evidence import fails validation, THEN THE Run_Registry SHALL preserve the registry and every existing artifact in the failure-before state.

### Requirement 2: Build Immutable Dataset and Sample Manifests

**User Story:** As an experiment owner, I want each dataset version represented by an immutable sample manifest, so that every result can be traced to exact samples and labels.

#### Acceptance Criteria

1. WHEN the Dataset_Catalog materializes a Sample_Manifest, THE Dataset_Catalog SHALL assign every sample exactly one Sample_ID.
2. WHEN the Dataset_Catalog materializes a Sample_Manifest, THE Dataset_Catalog SHALL make the Sample_ID set equal to the set of materialized dataset samples.
3. WHEN the Dataset_Catalog materializes a grouped-task Sample_Manifest, THE Dataset_Catalog SHALL assign every sample exactly one non-empty Group_ID.
4. WHEN the Dataset_Catalog materializes a Sample_Manifest, THE Dataset_Catalog SHALL store an input content hash for every required input of every sample.
5. WHEN the Dataset_Catalog materializes a Sample_Manifest, THE Dataset_Catalog SHALL store label-schema metadata for every sample.
6. WHEN the Dataset_Catalog computes Canonical_Order, THE Dataset_Catalog SHALL sort the complete Sample_ID set by the normalized Sample_ID byte sequence.
7. WHEN the same dataset content is enumerated in different input orders, THE Dataset_Catalog SHALL produce identical Canonical_Order values.
8. WHEN the same dataset content is enumerated in different input orders, THE Dataset_Catalog SHALL produce identical Dataset_Hash values.
9. WHEN the same Canonical_Serialization is hashed more than once, THE Dataset_Catalog SHALL produce the same Dataset_Hash.
10. IF the observed sample count differs from the registered expected sample count, THEN THE Dataset_Catalog SHALL block the formal benchmark before Protocol_Manifest creation.
11. IF the observed sample count differs from the registered expected sample count, THEN THE Dataset_Catalog SHALL emit an Error_Record containing `sample_count`, the expected count, and the observed count.
12. IF a Sample_ID is duplicated, THEN THE Dataset_Catalog SHALL block the formal benchmark before Protocol_Manifest creation.
13. IF a Sample_ID is duplicated, THEN THE Dataset_Catalog SHALL emit an Error_Record containing every duplicated Sample_ID.
14. IF a required Group_ID is absent, THEN THE Dataset_Catalog SHALL block the formal benchmark before Protocol_Manifest creation.
15. IF a required Group_ID is absent, THEN THE Dataset_Catalog SHALL emit an Error_Record containing every affected Sample_ID and the field `group_id`.
16. IF a label dimension is invalid, THEN THE Dataset_Catalog SHALL block the formal benchmark before Protocol_Manifest creation.
17. IF a label dimension is invalid, THEN THE Dataset_Catalog SHALL emit an Error_Record containing every affected Sample_ID, the expected dimension, and the observed dimension.
18. IF a required input is unreadable, THEN THE Dataset_Catalog SHALL block the formal benchmark before Protocol_Manifest creation.
19. IF a required input is unreadable, THEN THE Dataset_Catalog SHALL emit an Error_Record containing every affected Sample_ID, input reference, and read error code.
20. IF Sample_Manifest validation fails, THEN THE Dataset_Catalog SHALL leave every previously completed Sample_Manifest and artifact unchanged.
21. WHEN dataset samples change, THE Dataset_Catalog SHALL create a new Sample_Manifest with a new Dataset_Hash.
22. WHEN dataset labels change, THE Dataset_Catalog SHALL create a new Sample_Manifest with a new Dataset_Hash.
23. WHEN a new dataset version is created, THE Dataset_Catalog SHALL leave every prior Sample_Manifest byte-for-byte unchanged.
24. WHEN the Histology Sample_Manifest is exported for publication, THE Dataset_Catalog SHALL replace each patient identifier with one stable de-identified Group_ID.
25. WHEN the same Histology Sample_Manifest is exported more than once, THE Dataset_Catalog SHALL produce the same patient-to-Group_ID mapping.

### Requirement 3: Plan Reusable and Leakage-Safe Split Manifests

**User Story:** As a reviewer, I want splits frozen independently of models, so that model comparisons cannot benefit from hidden data access or split drift.

#### Acceptance Criteria

1. WHEN a benchmark request is planned, THE Protocol_Planner SHALL persist the Protocol_Manifest before the Model_Registry builds a model.
2. WHEN the Protocol_Planner creates a fold, THE Protocol_Planner SHALL store the complete Sample_ID set of every split in Canonical_Order.
3. WHEN the Protocol_Planner creates an Outer_Fold, THE Protocol_Planner SHALL make the intersection of Outer_Train_Set Sample_IDs and Outer_Test_Set Sample_IDs equal to the empty set.
4. WHEN the Protocol_Planner creates an Outer_Fold, THE Protocol_Planner SHALL make the union of Outer_Train_Set Sample_IDs and Outer_Test_Set Sample_IDs equal to the fold's expected Sample_ID set.
5. WHEN the Protocol_Planner creates a grouped Outer_Fold, THE Protocol_Planner SHALL make the intersection of Outer_Train_Set Group_IDs and Outer_Test_Set Group_IDs equal to the empty set.
6. WHEN the Protocol_Planner creates an Inner_Fold, THE Protocol_Planner SHALL make the union of inner training Sample_IDs and inner validation Sample_IDs a subset of the corresponding Outer_Train_Set.
7. WHEN the Protocol_Planner creates an Inner_Fold, THE Protocol_Planner SHALL make the intersection of inner training Sample_IDs and inner validation Sample_IDs equal to the empty set.
8. WHEN two models are compared under one protocol, THE Benchmark_System SHALL provide both models the same Protocol_Hash.
9. WHEN two models are compared under one protocol, THE Benchmark_System SHALL provide both models identical fold Sample_ID sets.
10. IF sample disjointness validation fails, THEN THE Protocol_Planner SHALL block execution before preprocessing or training.
11. IF sample disjointness validation fails, THEN THE Protocol_Planner SHALL emit an Error_Record containing the Fold_ID and intersecting Sample_IDs.
12. IF group disjointness validation fails, THEN THE Protocol_Planner SHALL block execution before preprocessing or training.
13. IF group disjointness validation fails, THEN THE Protocol_Planner SHALL emit an Error_Record containing the Fold_ID and intersecting Group_IDs.
14. IF expected coverage validation fails, THEN THE Protocol_Planner SHALL block execution before preprocessing or training.
15. IF expected coverage validation fails, THEN THE Protocol_Planner SHALL emit an Error_Record containing the Fold_ID, missing Sample_IDs, and unexpected Sample_IDs.
16. IF inner containment validation fails, THEN THE Protocol_Planner SHALL block execution before preprocessing or training.
17. IF inner containment validation fails, THEN THE Protocol_Planner SHALL emit an Error_Record containing the Fold_ID and out-of-bound Sample_IDs.
18. IF Protocol_Manifest validation fails, THEN THE Protocol_Planner SHALL leave every existing Protocol_Manifest and artifact unchanged.
19. WHEN a Protocol_Manifest is serialized and reloaded, THE Protocol_Planner SHALL preserve exact equality of every fold membership set.
20. WHEN a Protocol_Manifest is serialized and reloaded, THE Protocol_Planner SHALL preserve the Protocol_Hash.
21. WHEN equivalent fold sets are supplied in different enumeration orders, THE Protocol_Planner SHALL produce the same Canonical_Serialization and Protocol_Hash.

### Requirement 4: Enforce Fold-Scoped Preprocessing

**User Story:** As a method evaluator, I want every learned transformation fitted only on permitted training samples, so that preprocessing cannot leak validation or test information.

#### Acceptance Criteria

1. WHEN the Preprocessing_Manager fits target normalization, THE Preprocessing_Manager SHALL record the exact Fit_Sample_IDs in the Fitted_Artifact.
2. WHEN the Preprocessing_Manager fits PCA, THE Preprocessing_Manager SHALL record the exact Fit_Sample_IDs in the Fitted_Artifact.
3. WHEN the Preprocessing_Manager fits band normalization, THE Preprocessing_Manager SHALL record the exact Fit_Sample_IDs in the Fitted_Artifact.
4. WHEN the Preprocessing_Manager fits imputation, THE Preprocessing_Manager SHALL record the exact Fit_Sample_IDs in the Fitted_Artifact.
5. WHEN the Preprocessing_Manager fits class weights, THE Preprocessing_Manager SHALL record the exact Fit_Sample_IDs in the Fitted_Artifact.
6. WHEN the Preprocessing_Manager fits a threshold, THE Preprocessing_Manager SHALL record the exact Fit_Sample_IDs in the Fitted_Artifact.
7. WHEN the Preprocessing_Manager fits calibration, THE Preprocessing_Manager SHALL record the exact Fit_Sample_IDs in the Fitted_Artifact.
8. WHEN a Fitted_Artifact is used for outer evaluation, THE Preprocessing_Manager SHALL verify that Fit_Sample_IDs are a subset of the corresponding Outer_Train_Set.
9. WHEN a candidate is evaluated in an Inner_Fold, THE Preprocessing_Manager SHALL make Fit_Sample_IDs equal to a subset of the inner training Sample_IDs.
10. IF Fit_Sample_IDs intersect the corresponding Outer_Test_Set, THEN THE Benchmark_System SHALL block the run before applying the Fitted_Artifact.
11. IF Fit_Sample_IDs intersect the corresponding Outer_Test_Set, THEN THE Claim_Gate SHALL assign the `invalid_or_leaky` Claim_Status.
12. IF Fit_Sample_IDs intersect the corresponding Outer_Test_Set, THEN THE Benchmark_System SHALL emit an Error_Record containing the Fitted_Artifact identifier, Fold_ID, and intersecting Sample_IDs.
13. IF Fit_Sample_IDs validation fails, THEN THE Benchmark_System SHALL leave the checkpoint, Prediction_Artifact, and previously completed artifacts unchanged.
14. WHEN HyperLeaf target normalization is fitted, THE Preprocessing_Manager SHALL make Fit_Sample_IDs equal to the permitted training Sample_ID set.
15. WHEN HyperLeaf target normalization is fitted, THE Preprocessing_Manager SHALL exclude validation targets from the target-statistics input set.
16. WHEN HyperLeaf target normalization is fitted, THE Preprocessing_Manager SHALL exclude Official_Test_Set targets from the target-statistics input set.
17. WHEN a PCA Representation_Key is requested, THE Preprocessing_Manager SHALL bind the PCA basis to the Dataset_Hash.
18. WHEN a PCA Representation_Key is requested, THE Preprocessing_Manager SHALL bind the PCA basis to the Protocol_Hash.
19. WHEN a PCA Representation_Key is requested, THE Preprocessing_Manager SHALL bind the PCA basis to the Fold_ID.
20. WHEN a PCA Representation_Key is requested, THE Preprocessing_Manager SHALL bind the PCA basis to the exact Fit_Sample_IDs.
21. WHEN a PCA Representation_Key is requested, THE Preprocessing_Manager SHALL bind the PCA basis to the preprocessing configuration hash.
22. WHEN a deterministic transformation is cached across folds, THE Feature_Cache SHALL include every fixed transformation parameter in Lineage.
23. IF Feature_Cache Lineage has a mismatched source hash, THEN THE Feature_Cache SHALL reject the cache entry before returning cached data.
24. IF Feature_Cache Lineage has a mismatched transform hash, THEN THE Feature_Cache SHALL reject the cache entry before returning cached data.
25. IF Feature_Cache Lineage has an invalid Fit_Sample_IDs boundary, THEN THE Feature_Cache SHALL reject the cache entry before returning cached data.
26. IF Feature_Cache Lineage has a mismatched tensor shape, THEN THE Feature_Cache SHALL reject the cache entry before returning cached data.
27. IF Feature_Cache validation fails, THEN THE Feature_Cache SHALL emit an Error_Record containing the cache key, failing field, expected value, and observed value.
28. IF Feature_Cache validation fails, THEN THE Feature_Cache SHALL leave the rejected entry and every completed artifact unchanged.
29. WHEN identical input hashes, preprocessing configuration, and seed are used, THE Preprocessing_Manager SHALL produce identical Fitted_Artifact state hashes.

### Requirement 5: Guarantee Selection and Outer-Test Isolation

**User Story:** As a scientific reviewer, I want configuration selection isolated from outer labels, so that reported performance estimates generalization rather than test-set adaptation.

#### Acceptance Criteria

1. WHEN the Training_Engine selects an epoch, THE Training_Engine SHALL use only Inner_Fold predictions from the corresponding Outer_Train_Set.
2. WHEN the Training_Engine selects a model family, THE Training_Engine SHALL use only Inner_Fold predictions from the corresponding Outer_Train_Set.
3. WHEN the Training_Engine selects a target route, THE Training_Engine SHALL use only Inner_Fold predictions from the corresponding Outer_Train_Set.
4. WHEN the Training_Engine selects a blend weight, THE Training_Engine SHALL use only Inner_Fold predictions from the corresponding Outer_Train_Set.
5. WHEN the Training_Engine selects a threshold, THE Training_Engine SHALL use only Inner_Fold predictions from the corresponding Outer_Train_Set.
6. WHEN the Training_Engine selects a calibration configuration, THE Training_Engine SHALL use only Inner_Fold predictions from the corresponding Outer_Train_Set.
7. WHEN a Selection_Decision is created, THE Training_Engine SHALL store the complete candidate set.
8. WHEN a Selection_Decision is created, THE Training_Engine SHALL store every candidate score.
9. WHEN a Selection_Decision is created, THE Training_Engine SHALL store the exact source Sample_ID set.
10. WHEN a Selection_Decision is created, THE Training_Engine SHALL store the metric direction.
11. WHEN a Selection_Decision is created, THE Training_Engine SHALL store the predeclared tie-break rule.
12. WHEN a Selection_Decision is created, THE Training_Engine SHALL store the source prediction hash.
13. WHEN a Selection_Decision is finalized, THE Training_Engine SHALL freeze the Selection_Decision before Outer_Test_Set labels become accessible.
14. WHEN the Training_Engine refits a selected configuration, THE Training_Engine SHALL apply the unchanged frozen Selection_Decision to the full Outer_Train_Set.
15. IF a Selection_Decision references an Outer_Test_Set Sample_ID, THEN THE Benchmark_System SHALL block the run before checkpoint refitting.
16. IF a Selection_Decision references an Outer_Test_Set Sample_ID, THEN THE Claim_Gate SHALL assign the `invalid_or_leaky` Claim_Status.
17. IF a Selection_Decision references an Outer_Test_Set Sample_ID, THEN THE Benchmark_System SHALL emit an Error_Record containing the Selection_Decision identifier, Fold_ID, and forbidden Sample_IDs.
18. IF Selection_Decision validation fails, THEN THE Benchmark_System SHALL leave every existing checkpoint, Prediction_Artifact, and completed artifact unchanged.
19. WHEN an outer checkpoint is frozen, THE Benchmark_System SHALL evaluate the Outer_Test_Set with gradient updates disabled.
20. WHEN an outer checkpoint is frozen, THE Benchmark_System SHALL evaluate the Outer_Test_Set with fitted-state updates disabled.
21. WHEN an official test evaluation succeeds, THE Benchmark_System SHALL increment the consumed Evaluation_Budget exactly once.
22. IF an official test evaluation request would exceed the Evaluation_Budget, THEN THE Benchmark_System SHALL reject the request before loading Official_Test_Set labels.
23. IF an official test evaluation request would exceed the Evaluation_Budget, THEN THE Benchmark_System SHALL emit an Error_Record containing the Protocol_Key, configured budget, consumed count, and requested count.
24. IF an official test evaluation request would exceed the Evaluation_Budget, THEN THE Benchmark_System SHALL leave every completed official Prediction_Artifact unchanged.

### Requirement 6: Register Stable Model Identity, Input Contracts, and Provenance

**User Story:** As a benchmark maintainer, I want stable model identities and complete provenance, so that names, implementations, and upstream sources cannot drift between runs and reports.

#### Acceptance Criteria

1. WHEN the Model_Registry resolves a Model_Key, THE Model_Registry SHALL return exactly one Model_Identity.
2. WHEN the Model_Registry resolves a Model_Key, THE Model_Registry SHALL return exactly one Input_Contract.
3. WHEN a model instance is built, THE Model_Registry SHALL verify equality between the instance class fully qualified name and the Model_Identity class fully qualified name.
4. WHEN a model run starts, THE Model_Registry SHALL record the model source revision.
5. WHEN a model run starts, THE Model_Registry SHALL record the configuration hash.
6. WHEN a model run starts, THE Model_Registry SHALL record the parameter count.
7. WHEN a model run starts, THE Model_Registry SHALL record the Representation_Key.
8. WHEN a model run starts, THE Model_Registry SHALL record the preprocessing key.
9. WHEN a model run starts, THE Model_Registry SHALL verify equality between supplied modalities and Input_Contract modalities.
10. WHEN a model run starts, THE Model_Registry SHALL verify equality between supplied tensor shapes and Input_Contract tensor shapes.
11. WHEN a model run starts, THE Model_Registry SHALL verify equality between supplied dtypes and Input_Contract dtypes.
12. WHEN a model run starts, THE Model_Registry SHALL verify equality between supplied channel or band order and the Input_Contract order.
13. WHEN a model run starts, THE Model_Registry SHALL verify equality between the supplied preprocessing key and the Input_Contract preprocessing key.
14. WHEN an Official_Model is registered, THE Model_Registry SHALL record the upstream repository URL.
15. WHEN an Official_Model is registered, THE Model_Registry SHALL record the upstream commit.
16. WHEN an Official_Model is registered, THE Model_Registry SHALL record the license reference.
17. WHEN an Official_Model is registered, THE Model_Registry SHALL record the source hash.
18. WHEN an Official_Model is registered, THE Model_Registry SHALL record the local patch hash.
19. WHEN an Official_Model is registered, THE Model_Registry SHALL record the upstream reproduction status.
20. WHEN an Adapted_Official_Model is registered, THE Model_Registry SHALL record every input adaptation.
21. WHEN an Adapted_Official_Model is registered, THE Model_Registry SHALL record every ignored modality.
22. WHEN a Proxy_Model is registered, THE Model_Registry SHALL include the word `proxy` in the display name.
23. WHEN a Proxy_Model is registered, THE Claim_Gate SHALL assign the `proxy_only` Claim_Status ceiling.
24. IF a Model_Key disagrees with the built model, THEN THE Model_Registry SHALL block the run before training.
25. IF a class fully qualified name disagrees with the built model, THEN THE Model_Registry SHALL block the run before training.
26. IF an implementation category disagrees with the built model, THEN THE Model_Registry SHALL block the run before training.
27. IF a Provenance field fails hash or equality validation, THEN THE Model_Registry SHALL block the run before training.
28. IF an Input_Contract field fails equality validation, THEN THE Model_Registry SHALL block the run before training.
29. IF model validation fails, THEN THE Model_Registry SHALL emit an Error_Record containing the Model_Key, failing field, expected value, and observed value.
30. IF model validation fails, THEN THE Model_Registry SHALL leave every checkpoint, Prediction_Artifact, and completed Run_Record unchanged.
31. WHEN a CLI selects a model, THE Benchmark_System SHALL resolve the model by Model_Key.

### Requirement 7: Maintain Benchmark Tiers and Matched Controls

**User Story:** As a paper reader, I want classic, in-repository, official, adapted, and proxy models clearly separated, so that unlike implementations are not presented as equivalent evidence.

#### Acceptance Criteria

1. WHEN a benchmark table is generated, THE Report_Generator SHALL place Classic_Model rows in a distinct implementation-category group.
2. WHEN a benchmark table is generated, THE Report_Generator SHALL place In_Repo_Model rows in a distinct implementation-category group.
3. WHEN a benchmark table is generated, THE Report_Generator SHALL place Official_Model rows in a distinct implementation-category group.
4. WHEN a benchmark table is generated, THE Report_Generator SHALL place Adapted_Official_Model rows in a distinct implementation-category group.
5. WHEN a benchmark table is generated, THE Report_Generator SHALL place Proxy_Model rows in a distinct implementation-category group.
6. WHEN a candidate uses a distinct Representation_Key, THE Benchmark_System SHALL require at least one Matched_Input_Control before permitting `primary_strict` Claim_Status.
7. WHEN model results are compared, THE Report_Generator SHALL display the Protocol_Key.
8. WHEN model results are compared, THE Report_Generator SHALL display the Representation_Key.
9. WHEN model results are compared, THE Report_Generator SHALL display the parameter count.
10. WHEN model results are compared, THE Report_Generator SHALL display the completed training steps.
11. WHEN model results are compared, THE Report_Generator SHALL display the input resolution.
12. WHEN model results are compared, THE Report_Generator SHALL display the pretraining data status.
13. WHEN model results are compared, THE Report_Generator SHALL display the wall-clock runtime.
14. WHEN model results are compared, THE Report_Generator SHALL display the peak device memory.
15. IF two results have different Protocol_Hash values, THEN THE Metric_Engine SHALL omit a paired rank value.
16. IF two results have different Protocol_Hash values, THEN THE Metric_Engine SHALL omit an improvement value.
17. WHEN results with different Protocol_Hash values appear in one report, THE Report_Generator SHALL place the results in separate protocol sections.
18. WHEN results with different Protocol_Hash values appear in one report, THE Report_Generator SHALL display a machine-readable non-comparability reason.
19. WHEN a Remote_Sensing_Sanity_Track result is reported, THE Report_Generator SHALL exclude the result from HyperLeaf2024 performance rankings.
20. WHEN a Remote_Sensing_Sanity_Track result is reported, THE Report_Generator SHALL exclude the result from HistologyHSI-GB performance rankings.
21. WHEN an Official_Model adaptation is requested, THE Reproduction_Gate SHALL require a passing upstream reproduction record before formal project comparison.
22. IF the upstream reproduction record is absent, THEN THE Reproduction_Gate SHALL block the adaptation before formal project comparison.
23. IF the upstream reproduction record is absent, THEN THE Reproduction_Gate SHALL emit an Error_Record containing the Model_Key and field `upstream_reproduction_record`.
24. IF the upstream reproduction record is absent, THEN THE Reproduction_Gate SHALL leave registered upstream artifacts and completed runs unchanged.

### Requirement 8: Execute HyperLeaf Official and LOVO Protocols

**User Story:** As a plant-phenotyping researcher, I want external-test and unseen-variety evaluations kept distinct, so that HyperLeaf claims reflect the intended generalization setting.

#### Acceptance Criteria

1. WHEN the HyperLeaf_Official_Protocol is planned, THE Protocol_Planner SHALL make the internal-selection sample set equal to the 1,590 labeled training samples.
2. WHEN the HyperLeaf_Official_Protocol is planned, THE Protocol_Planner SHALL make the Official_Test_Set equal to the 820 official test samples.
3. WHEN the HyperLeaf_Official_Protocol is planned, THE Protocol_Planner SHALL make the intersection of the 1,590 training Sample_IDs and 820 Official_Test_Set Sample_IDs equal to the empty set.
4. WHEN the HyperLeaf_Official_Protocol selects a configuration, THE Training_Engine SHALL use only grouped Inner_Fold evidence from the 1,590 labeled training samples.
5. WHEN the HyperLeaf_Official_Protocol selects an epoch policy, THE Training_Engine SHALL use only grouped Inner_Fold evidence from the 1,590 labeled training samples.
6. WHEN the HyperLeaf_Official_Protocol completes selection, THE Training_Engine SHALL freeze the selected configuration before Official_Test_Set labels become accessible.
7. WHEN the HyperLeaf_Official_Protocol completes selection, THE Training_Engine SHALL freeze the epoch policy before Official_Test_Set labels become accessible.
8. WHEN the HyperLeaf_Official_Protocol completes selection, THE Training_Engine SHALL refit the frozen configuration using all 1,590 labeled training samples.
9. WHEN a HyperLeaf_Official_Protocol result is reported, THE Metric_Engine SHALL report GrainWeight R².
10. WHEN a HyperLeaf_Official_Protocol result is reported, THE Metric_Engine SHALL report Gsw R².
11. WHEN a HyperLeaf_Official_Protocol result is reported, THE Metric_Engine SHALL report PhiPS2 R².
12. WHEN a HyperLeaf_Official_Protocol result is reported, THE Metric_Engine SHALL report mean R².
13. WHEN a HyperLeaf_Official_Protocol result is reported, THE Metric_Engine SHALL report MAE.
14. WHEN the HyperLeaf_LOVO_Protocol is planned, THE Protocol_Planner SHALL make each Outer_Test_Set Group_ID set equal to one complete wheat variety.
15. WHEN the HyperLeaf_LOVO_Protocol is planned, THE Protocol_Planner SHALL make the outer-test Group_ID sets pairwise disjoint.
16. WHEN the HyperLeaf_LOVO_Protocol is planned, THE Protocol_Planner SHALL make the union of outer-test Group_ID sets equal to the complete wheat-variety Group_ID set.
17. WHEN a HyperLeaf_LOVO_Protocol candidate route is selected, THE Training_Engine SHALL use only Inner_Fold varieties within the corresponding Outer_Train_Set.
18. WHEN a HyperLeaf_LOVO_Protocol epoch is selected, THE Training_Engine SHALL use only Inner_Fold varieties within the corresponding Outer_Train_Set.
19. WHEN a HyperLeaf_LOVO_Protocol Selection_Decision is frozen, THE Training_Engine SHALL exclude the held-out wheat variety from every selection source Sample_ID.
20. WHEN HyperLeaf_LOVO_Protocol results are aggregated, THE Metric_Engine SHALL report per-fold metrics as separately named values.
21. WHEN HyperLeaf_LOVO_Protocol results are aggregated, THE Metric_Engine SHALL report per-target metrics as separately named values.
22. WHEN HyperLeaf_LOVO_Protocol results are aggregated, THE Metric_Engine SHALL report fold-macro metrics as separately named values.
23. WHEN HyperLeaf_LOVO_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF metrics as separately named values.
24. WHEN HyperLeaf random 80/20 results are reported, THE Claim_Gate SHALL assign the `development` Claim_Status ceiling.
25. WHEN HyperLeaf official and LOVO results appear in one document, THE Report_Generator SHALL place the results in separate protocol sections.
26. WHEN HyperLeaf official and random-split results appear in one document, THE Report_Generator SHALL place the results in separate protocol sections.
27. WHEN HyperLeaf LOVO and random-split results appear in one document, THE Report_Generator SHALL place the results in separate protocol sections.
28. IF HyperLeaf protocol isolation validation fails, THEN THE Benchmark_System SHALL block evaluation before reading the held-out labels.
29. IF HyperLeaf protocol isolation validation fails, THEN THE Benchmark_System SHALL emit an Error_Record containing the Protocol_Key, Fold_ID, and forbidden Sample_IDs or Group_IDs.
30. IF HyperLeaf protocol isolation validation fails, THEN THE Benchmark_System SHALL leave completed HyperLeaf artifacts unchanged.

### Requirement 9: Execute Histology Nested Patient CV and Fixed-Epoch LOPO

**User Story:** As a biomedical reviewer, I want patient-isolated validation with patient-level uncertainty, so that ROI correlation does not inflate evidence.

#### Acceptance Criteria

1. WHEN the Histology_Nested_Patient_Protocol is planned, THE Protocol_Planner SHALL make the intersection of outer training Group_IDs and outer test Group_IDs equal to the empty set.
2. WHEN the Histology_Nested_Patient_Protocol is planned, THE Protocol_Planner SHALL make the intersection of inner training Group_IDs and inner validation Group_IDs equal to the empty set.
3. WHEN the Histology_Nested_Patient_Protocol is planned, THE Protocol_Planner SHALL assign every patient Group_ID entirely to one side of each outer split.
4. WHEN the Histology_Nested_Patient_Protocol is planned, THE Protocol_Planner SHALL assign every patient Group_ID entirely to one side of each inner split.
5. WHEN a Histology_Nested_Patient_Protocol configuration is selected, THE Training_Engine SHALL use only inner patient groups from the corresponding Outer_Train_Set.
6. WHEN a Histology_Nested_Patient_Protocol epoch is selected, THE Training_Engine SHALL use only inner patient groups from the corresponding Outer_Train_Set.
7. WHEN a Histology_Nested_Patient_Protocol Outer_Fold is evaluated, THE Benchmark_System SHALL require a frozen Selection_Decision before outer patient labels become accessible.
8. WHEN a Histology_Nested_Patient_Protocol Outer_Fold is evaluated, THE Benchmark_System SHALL consume exactly one outer evaluation for the frozen configuration.
9. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report fold-macro accuracy as a separately named value.
10. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF accuracy as a separately named value.
11. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF balanced accuracy as a separately named value.
12. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF AUROC as a separately named value.
13. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF sensitivity as a separately named value.
14. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF specificity as a separately named value.
15. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report the pooled OOF confusion matrix as a separately named value.
16. WHEN Histology_Nested_Patient_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF ECE as a separately named value.
17. WHEN the Histology_Fixed_Epoch_LOPO_Protocol is planned, THE Protocol_Planner SHALL freeze the epoch policy before any held-out patient labels become accessible.
18. WHEN the Histology_Fixed_Epoch_LOPO_Protocol is planned, THE Protocol_Planner SHALL make each Outer_Test_Set Group_ID set equal to one patient Group_ID.
19. WHEN the Histology_Fixed_Epoch_LOPO_Protocol is planned, THE Protocol_Planner SHALL make the union of held-out patient Group_IDs equal to the complete patient Group_ID set.
20. WHEN Histology_Fixed_Epoch_LOPO_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF accuracy.
21. WHEN Histology_Fixed_Epoch_LOPO_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF balanced accuracy.
22. WHEN Histology_Fixed_Epoch_LOPO_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF AUROC.
23. WHEN Histology_Fixed_Epoch_LOPO_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF sensitivity.
24. WHEN Histology_Fixed_Epoch_LOPO_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF specificity.
25. WHEN Histology_Fixed_Epoch_LOPO_Protocol results are aggregated, THE Metric_Engine SHALL report the pooled OOF confusion matrix.
26. WHEN Histology_Fixed_Epoch_LOPO_Protocol results are aggregated, THE Metric_Engine SHALL report pooled OOF ECE.
27. WHEN a Histology bootstrap confidence interval is computed, THE Metric_Engine SHALL use patient Group_ID as the resampling unit.
28. WHEN a Histology bootstrap confidence interval is computed, THE Metric_Engine SHALL use exactly 10,000 resamples.
29. WHEN a Histology bootstrap confidence interval is computed, THE Metric_Engine SHALL report a two-sided 95% confidence interval.
30. WHEN a Histology paired comparison is computed, THE Metric_Engine SHALL resample paired patient Group_ID contributions.
31. WHEN a Histology paired comparison is computed, THE Metric_Engine SHALL use exactly 10,000 resamples.
32. WHEN a Histology paired comparison is computed, THE Metric_Engine SHALL report a two-sided 95% confidence interval.
33. WHEN a Histology threshold is fitted, THE Preprocessing_Manager SHALL use only Inner_Fold OOF_Prediction data.
34. WHEN a Histology temperature is fitted, THE Preprocessing_Manager SHALL use only Inner_Fold OOF_Prediction data.
35. WHEN Histology primary metrics are reported, THE Metric_Engine SHALL retain results at the fixed probability threshold `0.5`.
36. WHEN an inner-fitted threshold result is reported, THE Metric_Engine SHALL label the result separately from the fixed-threshold `0.5` result.
37. WHEN cached spatial-mean PCA results are reported, THE Report_Generator SHALL assign a Representation_Key distinct from Full_Cube spatial PCA.
38. WHEN Full_Cube spatial PCA results are reported, THE Report_Generator SHALL assign a Representation_Key distinct from cached spatial-mean PCA.
39. IF Histology patient isolation validation fails, THEN THE Benchmark_System SHALL block evaluation before reading held-out patient labels.
40. IF Histology patient isolation validation fails, THEN THE Benchmark_System SHALL emit an Error_Record containing the Protocol_Key, Fold_ID, and intersecting patient Group_IDs.
41. IF Histology patient isolation validation fails, THEN THE Benchmark_System SHALL leave completed Histology artifacts unchanged.

### Requirement 10: Govern Model Iteration and Candidate Promotion

**User Story:** As a research lead, I want candidate changes promoted through predefined evidence gates, so that iteration does not turn outer-test feedback into model selection.

#### Acceptance Criteria

1. WHEN a new model candidate is proposed, THE Benchmark_System SHALL register the candidate selection space before outer evaluation.
2. WHEN a new representation candidate is proposed, THE Benchmark_System SHALL register the candidate selection space before outer evaluation.
3. WHEN a new loss candidate is proposed, THE Benchmark_System SHALL register the candidate selection space before outer evaluation.
4. WHEN a new augmentation candidate is proposed, THE Benchmark_System SHALL register the candidate selection space before outer evaluation.
5. WHEN a new target-route candidate is proposed, THE Benchmark_System SHALL register the candidate selection space before outer evaluation.
6. WHEN a new Inference_Branch candidate is proposed, THE Benchmark_System SHALL register the candidate selection space before outer evaluation.
7. WHEN a new calibration candidate is proposed, THE Benchmark_System SHALL register the candidate selection space before outer evaluation.
8. WHEN the HyperLeaf target-routed RGB/PCA consistency candidate selects target routes, THE Training_Engine SHALL use only Inner_Fold variety evidence.
9. WHEN the HyperLeaf target-routed RGB/PCA consistency candidate selects consistency settings, THE Training_Engine SHALL use only Inner_Fold variety evidence.
10. WHEN the Histology consistency-trained PCA-only candidate selects an Inference_Branch, THE Training_Engine SHALL use only Inner_Fold patient evidence.
11. WHEN the Histology consistency-trained PCA-only candidate completes selection, THE Training_Engine SHALL freeze PCA-only as the Inference_Branch before outer patient labels become accessible.
12. WHEN a candidate is compared with a control, THE Benchmark_System SHALL use the same Dataset_Hash for both results.
13. WHEN a candidate is compared with a control, THE Benchmark_System SHALL use the same Protocol_Hash for both results.
14. WHEN a candidate is compared with a control, THE Benchmark_System SHALL use the same Seed_Set for both results.
15. WHEN a candidate is compared with a control, THE Benchmark_System SHALL use identical fold manifests for both results.
16. WHEN a candidate is compared with a control, THE Benchmark_System SHALL use the same metric configuration for both results.
17. WHEN a candidate is compared with a control, THE Benchmark_System SHALL use the declared Resource_Budget for both results.
18. IF a candidate lacks the declared Seed_Set, THEN THE Claim_Gate SHALL assign the `strict_candidate` Claim_Status ceiling or a lower applicable ceiling.
19. IF a candidate lacks complete folds, THEN THE Claim_Gate SHALL assign the `strict_candidate` Claim_Status ceiling or a lower applicable ceiling.
20. IF a candidate lacks unified-runner reproduction, THEN THE Claim_Gate SHALL assign the `strict_candidate` Claim_Status ceiling or a lower applicable ceiling.
21. IF a candidate lacks a required Matched_Input_Control, THEN THE Claim_Gate SHALL assign the `strict_candidate` Claim_Status ceiling or a lower applicable ceiling.
22. WHEN a candidate underperforms, THE Run_Registry SHALL append the result as negative evidence with the tested configuration.
23. WHEN a candidate produces unstable independent-seed results, THE Run_Registry SHALL append the result as negative evidence with the tested configuration.
24. WHEN a candidate fails a representation ablation, THE Run_Registry SHALL append the result as negative evidence with the tested configuration.
25. WHEN negative evidence is appended, THE Run_Registry SHALL leave existing positive and negative evidence unchanged.
26. WHEN a candidate is promoted to `primary_strict`, THE Claim_Gate SHALL require a strict protocol.
27. WHEN a candidate is promoted to `primary_strict`, THE Claim_Gate SHALL require complete Provenance.
28. WHEN a candidate is promoted to `primary_strict`, THE Claim_Gate SHALL require the complete declared Seed_Set.
29. WHEN a candidate is promoted to `primary_strict`, THE Claim_Gate SHALL require complete folds.
30. WHEN a candidate is promoted to `primary_strict`, THE Claim_Gate SHALL require every required control.
31. WHEN a candidate is promoted to `primary_strict`, THE Claim_Gate SHALL require reproducible Prediction_Artifact files.
32. WHEN a Global Route A result is described, THE Report_Generator SHALL characterize the model as a global fusion baseline.
33. WHEN a Global Route A result is described, THE Report_Generator SHALL omit attribution to learnable single-token correspondence.
34. WHEN a Local Route A result lacks an attention-specific Matched_Input_Control, THE Report_Generator SHALL identify local tokenization, MIL pooling, and cross-modal interaction as the combined method.
35. IF candidate comparison validation fails, THEN THE Benchmark_System SHALL block promotion before changing Claim_Status.
36. IF candidate comparison validation fails, THEN THE Benchmark_System SHALL emit an Error_Record containing the candidate Run_ID, control Run_ID, failing comparison field, expected value, and observed value.
37. IF candidate comparison validation fails, THEN THE Benchmark_System SHALL leave existing Run_Record and Aggregate_Record values unchanged.

### Requirement 11: Compute Prediction-First Metrics and Statistics

**User Story:** As an auditor, I want every published number recomputable from per-sample predictions, so that checkpoint histories and copied summaries cannot become the sole source of evidence.

#### Acceptance Criteria

1. WHEN an Outer_Fold run completes, THE Benchmark_System SHALL create exactly one Prediction_Artifact record for every expected Outer_Test_Set Sample_ID.
2. WHEN an Outer_Fold run completes, THE Benchmark_System SHALL make the Prediction_Artifact Sample_ID set equal to the expected Outer_Test_Set Sample_ID set.
3. WHEN a Prediction_Artifact is validated, THE Benchmark_System SHALL verify uniqueness of every Prediction_Key.
4. IF an expected Sample_ID is absent from a Prediction_Artifact, THEN THE Benchmark_System SHALL reject the Prediction_Artifact before metric computation.
5. IF an unexpected Sample_ID appears in a Prediction_Artifact, THEN THE Benchmark_System SHALL reject the Prediction_Artifact before metric computation.
6. IF a duplicate Prediction_Key appears in a Prediction_Artifact, THEN THE Metric_Engine SHALL reject the affected aggregate before metric computation.
7. IF Prediction_Artifact completeness validation fails, THEN THE Benchmark_System SHALL emit an Error_Record containing missing Sample_IDs, unexpected Sample_IDs, and duplicate Prediction_Keys.
8. IF Prediction_Artifact completeness validation fails, THEN THE Benchmark_System SHALL leave every completed Prediction_Artifact and Aggregate_Record unchanged.
9. WHEN a formal metric is computed, THE Metric_Engine SHALL use only a validated Prediction_Artifact as metric input.
10. WHEN a metric is recomputed from the same Prediction_Artifact and metric configuration, THE Metric_Engine SHALL produce a point estimate with absolute difference no greater than the Numeric_Tolerance.
11. WHEN multiple prediction shards are merged in different orders, THE Metric_Engine SHALL produce identical aggregate membership.
12. WHEN multiple prediction shards are merged in different orders, THE Metric_Engine SHALL produce metric values with absolute differences no greater than the Numeric_Tolerance.
13. IF AUROC is undefined for a single-class fold, THEN THE Metric_Engine SHALL record a null metric value.
14. IF AUROC is undefined for a single-class fold, THEN THE Metric_Engine SHALL record the machine-readable reason `single_class_fold`.
15. IF R² is undefined for a zero-variance target, THEN THE Metric_Engine SHALL record a null metric value.
16. IF R² is undefined for a zero-variance target, THEN THE Metric_Engine SHALL record the machine-readable reason `zero_variance_target`.
17. WHEN seed variance is reported, THE Metric_Engine SHALL calculate seed variance across the declared Seed_Set while holding the aggregation policy fixed.
18. WHEN seed variance is reported, THE Metric_Engine SHALL label the value `seed_variance`.
19. WHEN fold variance is reported, THE Metric_Engine SHALL calculate fold variance across Fold_ID values while holding the aggregation policy fixed.
20. WHEN fold variance is reported, THE Metric_Engine SHALL label the value `fold_variance`.
21. WHEN an Aggregate_Record is created, THE Metric_Engine SHALL record the complete Member_Run_IDs set.
22. WHEN an Aggregate_Record is created, THE Metric_Engine SHALL record the content hash of every contributing Prediction_Artifact.
23. WHEN an Aggregate_Record is created, THE Metric_Engine SHALL make aggregate membership equal to the validated records referenced by Member_Run_IDs.
24. WHEN a bootstrap confidence interval is reported, THE Metric_Engine SHALL record the resampling unit.
25. WHEN a bootstrap confidence interval is reported, THE Metric_Engine SHALL record the number of resamples.
26. WHEN a bootstrap confidence interval is reported, THE Metric_Engine SHALL record the confidence level and sidedness.
27. WHEN a bootstrap confidence interval is reported, THE Metric_Engine SHALL record the random seed.
28. WHEN a paired comparison is reported, THE Metric_Engine SHALL record the resampling unit.
29. WHEN a paired comparison is reported, THE Metric_Engine SHALL record the number of resamples.
30. WHEN a paired comparison is reported, THE Metric_Engine SHALL record the confidence level and sidedness.
31. WHEN a paired comparison is reported, THE Metric_Engine SHALL record the random seed.

### Requirement 12: Synchronize Results from One Authoritative Source

**User Story:** As a paper author, I want JSON, CSV, Markdown, and audit outputs generated from the same records, so that names and numbers cannot drift across artifacts.

#### Acceptance Criteria

1. WHEN the Report_Generator creates a result table, THE Report_Generator SHALL read Run_Record data from the Run_Registry.
2. WHEN the Report_Generator creates a result table, THE Report_Generator SHALL read Aggregate_Record data from the Run_Registry.
3. WHEN one Aggregate_Record is rendered to JSON, CSV, and Markdown, THE Report_Generator SHALL use identical metric values in all three formats.
4. WHEN one Aggregate_Record is rendered to JSON, CSV, and Markdown, THE Report_Generator SHALL use identical Model_Identity values in all three formats.
5. WHEN one Aggregate_Record is rendered to JSON, CSV, and Markdown, THE Report_Generator SHALL use identical protocol identity values in all three formats.
6. WHEN one Aggregate_Record is rendered to JSON, CSV, and Markdown, THE Report_Generator SHALL use identical Claim_Status values in all three formats.
7. WHEN numeric publication cells are rendered to JSON, THE Report_Generator SHALL format every numeric publication cell to four decimal places.
8. WHEN numeric publication cells are rendered to CSV, THE Report_Generator SHALL format every numeric publication cell to four decimal places.
9. WHEN numeric publication cells are rendered to Markdown, THE Report_Generator SHALL format every numeric publication cell to four decimal places.
10. WHEN generated JSON, CSV, and Markdown formats are compared, THE Report_Generator SHALL make every displayed numeric cell exactly equal after four-decimal formatting.
11. IF an expected fold is missing, THEN THE Report_Generator SHALL mark the result as `INCOMPLETE`.
12. IF an expected fold is missing, THEN THE Report_Generator SHALL omit the Publication_Aggregate.
13. IF an expected fold is missing, THEN THE Report_Generator SHALL emit an Error_Record containing every missing Fold_ID.
14. IF a declared seed is missing, THEN THE Report_Generator SHALL mark the result as `INCOMPLETE`.
15. IF a declared seed is missing, THEN THE Report_Generator SHALL omit the Publication_Aggregate.
16. IF a declared seed is missing, THEN THE Report_Generator SHALL emit an Error_Record containing every missing seed.
17. WHEN a result is superseded, THE Report_Generator SHALL display the Superseding_Record identifier.
18. WHEN a result is superseded, THE Report_Generator SHALL retain a traceable reference to the prior record identifier.
19. WHEN a table includes a caveated result, THE Report_Generator SHALL render the machine-readable Claim_Status.
20. WHEN a table includes a caveated result, THE Report_Generator SHALL render the Claim_Gate explanation.
21. IF a hand-edited summary conflicts with the Run_Registry, THEN THE Report_Generator SHALL reject publication output before writing publication files.
22. IF a hand-edited summary conflicts with the Run_Registry, THEN THE Report_Generator SHALL emit an Error_Record containing every conflicting field, registry value, and summary value.
23. IF report generation fails validation, THEN THE Report_Generator SHALL leave every existing publication artifact unchanged.

### Requirement 13: Support Recoverable and Append-Only Execution

**User Story:** As an experiment operator, I want interrupted benchmark jobs resumed safely, so that long evaluations can recover without corrupting evidence or duplicating samples.

#### Acceptance Criteria

1. WHEN a benchmark job starts, THE Recovery_Manager SHALL assign a unique Run_ID.
2. WHEN a benchmark job starts, THE Recovery_Manager SHALL persist the Frozen_Run_Identity before training begins.
3. WHEN an artifact is written, THE Recovery_Manager SHALL write the artifact to a temporary path before publication.
4. WHEN a temporary artifact write completes, THE Recovery_Manager SHALL verify the artifact content hash before atomic publication.
5. WHEN an artifact is atomically published, THE Recovery_Manager SHALL mark the artifact complete exactly once.
6. IF a run is interrupted, THEN THE Run_Registry SHALL append a `FAILED` or `ABORTED` status record.
7. IF a run is interrupted, THEN THE Run_Registry SHALL record the last completed stage.
8. IF a run is interrupted, THEN THE Run_Registry SHALL leave every completed artifact unchanged.
9. WHEN an interrupted run resumes, THE Recovery_Manager SHALL verify equality of the Dataset_Hash with the Frozen_Run_Identity.
10. WHEN an interrupted run resumes, THE Recovery_Manager SHALL verify equality of the Protocol_Hash with the Frozen_Run_Identity.
11. WHEN an interrupted run resumes, THE Recovery_Manager SHALL verify equality of the Model_Key with the Frozen_Run_Identity.
12. WHEN an interrupted run resumes, THE Recovery_Manager SHALL verify equality of the configuration hash with the Frozen_Run_Identity.
13. WHEN an interrupted run resumes, THE Recovery_Manager SHALL verify equality of the Fold_ID with the Frozen_Run_Identity.
14. WHEN an interrupted run resumes, THE Recovery_Manager SHALL verify equality of the seed with the Frozen_Run_Identity.
15. WHEN an interrupted run resumes, THE Recovery_Manager SHALL verify Environment_Lock compatibility with the Frozen_Run_Identity.
16. WHEN all Frozen_Run_Identity checks pass, THE Recovery_Manager SHALL restore optimizer state from the last complete recovery artifact.
17. WHEN all Frozen_Run_Identity checks pass, THE Recovery_Manager SHALL restore scheduler state from the last complete recovery artifact.
18. WHEN all Frozen_Run_Identity checks pass, THE Recovery_Manager SHALL restore model state from the last complete recovery artifact.
19. WHEN all Frozen_Run_Identity checks pass, THE Recovery_Manager SHALL restore stage state from the last complete recovery artifact.
20. IF a resume Dataset_Hash differs from the Frozen_Run_Identity, THEN THE Recovery_Manager SHALL reject in-place resume before state restoration.
21. IF a resume Protocol_Hash differs from the Frozen_Run_Identity, THEN THE Recovery_Manager SHALL reject in-place resume before state restoration.
22. IF a resume Model_Key differs from the Frozen_Run_Identity, THEN THE Recovery_Manager SHALL reject in-place resume before state restoration.
23. IF a resume configuration hash differs from the Frozen_Run_Identity, THEN THE Recovery_Manager SHALL reject in-place resume before state restoration.
24. IF a resume Fold_ID differs from the Frozen_Run_Identity, THEN THE Recovery_Manager SHALL reject in-place resume before state restoration.
25. IF a resume seed differs from the Frozen_Run_Identity, THEN THE Recovery_Manager SHALL reject in-place resume before state restoration.
26. IF a resume Environment_Lock is incompatible with the Frozen_Run_Identity, THEN THE Recovery_Manager SHALL reject in-place resume before state restoration.
27. IF in-place resume is rejected, THEN THE Recovery_Manager SHALL require a new Run_ID.
28. IF in-place resume is rejected, THEN THE Recovery_Manager SHALL emit an Error_Record containing every mismatched frozen field, expected value, and observed value.
29. IF in-place resume is rejected, THEN THE Recovery_Manager SHALL leave every completed artifact unchanged.
30. WHEN an aggregate lacks declared jobs, THE Recovery_Manager SHALL schedule exactly the missing Fold_ID, Model_Key, seed, and configuration-hash combinations.
31. WHEN duplicate completed jobs exist, THE Run_Registry SHALL select at most one Canonical_Completed_Record for aggregation.
32. WHEN duplicate completed jobs exist, THE Run_Registry SHALL retain every duplicate completed record for audit.
33. WHEN a Legacy_Evidence source is imported, THE Run_Registry SHALL leave the source file unchanged.
34. WHEN a Legacy_Evidence source is imported, THE Run_Registry SHALL leave the source checkpoint unchanged.
35. IF a write target already contains a completed artifact, THEN THE Recovery_Manager SHALL reject the overwrite before opening the completed artifact for modification.
36. IF a write target already contains a completed artifact, THEN THE Recovery_Manager SHALL require a new artifact identifier or Run_ID.

### Requirement 14: Enforce Resource Budgets and Efficient Artifact Handling

**User Story:** As an experiment planner, I want compute and storage budgets declared and measured, so that model comparisons remain feasible and transparent.

#### Acceptance Criteria

1. WHEN a formal benchmark request is frozen, THE Benchmark_System SHALL record a Resource_Budget for every model and protocol combination.
2. WHEN a Resource_Budget value is declared, THE Benchmark_System SHALL require the value to be a finite positive integer in the declared unit.
3. WHEN a Resource_Budget limit is declared, THE Benchmark_System SHALL classify the limit as Hard_Limit or Advisory_Limit.
4. IF a Resource_Budget value is zero, negative, non-integer, or non-finite, THEN THE Benchmark_System SHALL reject the formal benchmark request before job scheduling.
5. IF Resource_Budget validation fails, THEN THE Benchmark_System SHALL emit an Error_Record containing the budget field, declared unit, and observed value.
6. IF Resource_Budget validation fails, THEN THE Benchmark_System SHALL leave existing Run_Record and artifact data unchanged.
7. WHEN a run executes, THE Training_Engine SHALL record completed epochs.
8. WHEN a run executes, THE Training_Engine SHALL record completed training steps.
9. WHEN a run executes, THE Training_Engine SHALL record wall-clock duration.
10. WHEN a run executes, THE Training_Engine SHALL record peak device memory.
11. WHEN a run executes, THE Training_Engine SHALL record device identity.
12. WHEN a run executes, THE Training_Engine SHALL record numeric precision.
13. WHEN a run executes, THE Training_Engine SHALL record parameter count.
14. WHEN a run executes, THE Training_Engine SHALL record input volume.
15. IF a run exceeds a Hard_Limit, THEN THE Training_Engine SHALL stop at the next checkpoint boundary.
16. IF a run exceeds a Hard_Limit, THEN THE Training_Engine SHALL record the budget field, limit, observed value, and stopping checkpoint.
17. IF a run exceeds an Advisory_Limit, THEN THE Training_Engine SHALL record a budget warning.
18. IF a run exceeds an Advisory_Limit, THEN THE Training_Engine SHALL preserve the configured execution policy without mandatory stopping.
19. WHEN raw hyperspectral cubes are read, THE Benchmark_System SHALL support chunked access.
20. WHEN raw hyperspectral cubes are read, THE Benchmark_System SHALL support memory-mapped access.
21. WHEN a Feature_Cache key is created for Fold_Scoped_Preprocessing output, THE Feature_Cache SHALL include the Dataset_Hash.
22. WHEN a Feature_Cache key is created for Fold_Scoped_Preprocessing output, THE Feature_Cache SHALL include the Protocol_Hash.
23. WHEN a Feature_Cache key is created for Fold_Scoped_Preprocessing output, THE Feature_Cache SHALL include the Fold_ID.
24. WHEN a Feature_Cache key is created for Fold_Scoped_Preprocessing output, THE Feature_Cache SHALL include the Representation_Key.
25. WHEN a Feature_Cache key is created for Fold_Scoped_Preprocessing output, THE Feature_Cache SHALL include the transform-state hash.
26. WHEN a deterministic raw-derived feature is reused, THE Feature_Cache SHALL verify equality between the current source content hash and Lineage source content hash.
27. IF a deterministic raw-derived feature source hash differs, THEN THE Feature_Cache SHALL reject reuse before returning cached data.
28. IF a deterministic raw-derived feature source hash differs, THEN THE Feature_Cache SHALL leave the existing cache entry unchanged.
29. WHEN checkpoints are retained under the default policy, THE Training_Engine SHALL preserve the selected checkpoint.
30. WHEN checkpoints are retained under the default policy, THE Training_Engine SHALL preserve the last checkpoint.
31. WHEN checkpoints are retained under the default policy, THE Training_Engine SHALL preserve the recovery state identified by the frozen configuration.
32. WHEN deterministic execution mode is selected, THE Benchmark_System SHALL record the selected mode and relevant framework determinism settings in the Run_Record.
33. WHEN performance execution mode is selected, THE Benchmark_System SHALL record the selected mode and relevant framework determinism settings in the Run_Record.

### Requirement 15: Gate CVPR 2027 Claims

**User Story:** As a CVPR 2027 author, I want every headline claim checked against explicit evidence rules, so that the paper distinguishes strict evidence from development and sanity results.

#### Acceptance Criteria

1. WHEN a CVPR_Claim is requested, THE Claim_Gate SHALL return exactly one Claim_Status.
2. WHEN a CVPR_Claim is requested, THE Claim_Gate SHALL return a machine-readable list of supporting reasons.
3. WHEN a CVPR_Claim is requested, THE Claim_Gate SHALL return a machine-readable list of blocking reasons.
4. WHEN multiple Claim_Status_Ceiling rules apply, THE Claim_Gate SHALL assign the lowest applicable Claim_Status_Ceiling.
5. IF evidence contains detected leakage, THEN THE Claim_Gate SHALL assign the `invalid_or_leaky` Claim_Status.
6. IF evidence contains a model identity mismatch, THEN THE Claim_Gate SHALL assign the `invalid_or_leaky` Claim_Status.
7. IF evidence uses a Proxy_Model, THEN THE Claim_Gate SHALL apply the `proxy_only` Claim_Status ceiling.
8. IF evidence belongs to the Remote_Sensing_Sanity_Track, THEN THE Claim_Gate SHALL apply the `sanity_only` Claim_Status ceiling.
9. IF evidence uses a random split, THEN THE Claim_Gate SHALL apply the `development` Claim_Status ceiling.
10. IF evidence uses outer data for model selection, THEN THE Claim_Gate SHALL apply the `development` Claim_Status ceiling or the `invalid_or_leaky` status when leakage is detected.
11. IF evidence is a single run, THEN THE Claim_Gate SHALL apply the `exploratory` Claim_Status ceiling.
12. IF evidence uses outer data for ensemble-weight search, THEN THE Claim_Gate SHALL apply the `exploratory` Claim_Status ceiling or the `invalid_or_leaky` status when leakage is detected.
13. IF evidence uses outer data for threshold search, THEN THE Claim_Gate SHALL apply the `exploratory` Claim_Status ceiling or the `invalid_or_leaky` status when leakage is detected.
14. IF strict evidence lacks complete Provenance, THEN THE Claim_Gate SHALL apply the `strict_candidate` Claim_Status ceiling.
15. IF strict evidence lacks a complete Seed_Set, THEN THE Claim_Gate SHALL apply the `strict_candidate` Claim_Status ceiling.
16. IF strict evidence lacks complete folds, THEN THE Claim_Gate SHALL apply the `strict_candidate` Claim_Status ceiling.
17. IF strict evidence lacks unified-runner reproduction, THEN THE Claim_Gate SHALL apply the `strict_candidate` Claim_Status ceiling.
18. IF strict evidence lacks a required Matched_Input_Control, THEN THE Claim_Gate SHALL apply the `strict_candidate` Claim_Status ceiling.
19. WHEN evidence satisfies the strict protocol check, THE Claim_Gate SHALL record the strict protocol check as passing.
20. WHEN evidence satisfies the Provenance check, THE Claim_Gate SHALL record the Provenance check as passing.
21. WHEN evidence satisfies the Seed_Set check, THE Claim_Gate SHALL record the Seed_Set check as passing.
22. WHEN evidence satisfies the fold completeness check, THE Claim_Gate SHALL record the fold completeness check as passing.
23. WHEN evidence satisfies the required-control check, THE Claim_Gate SHALL record the required-control check as passing.
24. WHEN evidence satisfies the Prediction_Artifact check, THE Claim_Gate SHALL record the Prediction_Artifact check as passing.
25. WHEN evidence satisfies the reproduction check, THE Claim_Gate SHALL record the reproduction check as passing.
26. WHEN every `primary_strict` prerequisite check passes, THE Claim_Gate SHALL permit the `primary_strict` Claim_Status.
27. WHEN a score increases without satisfying an additional claim prerequisite, THE Claim_Gate SHALL preserve the prior Claim_Status ceiling.
28. WHEN HyperLeaf2024 evidence is aggregated for reporting, THE Claim_Gate SHALL preserve a HyperLeaf2024 task-level conclusion.
29. WHEN HistologyHSI-GB evidence is aggregated for reporting, THE Claim_Gate SHALL preserve a HistologyHSI-GB task-level conclusion.
30. WHEN Remote_Sensing_Sanity_Track evidence is aggregated for reporting, THE Claim_Gate SHALL preserve a Remote_Sensing_Sanity_Track task-level conclusion.
31. WHEN evidence from multiple task tracks is reported, THE Claim_Gate SHALL omit a combined overall score.
32. WHEN a CVPR_Claim enters a headline table, THE Report_Generator SHALL include the protocol name.
33. WHEN a CVPR_Claim enters a headline table, THE Report_Generator SHALL include the uncertainty estimate.
34. WHEN a CVPR_Claim enters a headline table, THE Report_Generator SHALL include the evidence limitation.
35. WHEN a CVPR_Claim enters a headline table, THE Report_Generator SHALL include the Claim_Status.
36. WHEN a CVPR_Claim enters an abstract evidence list, THE Report_Generator SHALL include the protocol name, uncertainty estimate, evidence limitation, and Claim_Status.
37. IF claim validation fails before classification completes, THEN THE Claim_Gate SHALL emit an Error_Record containing the claim identifier, failing field, expected value, and observed value.
38. IF claim validation fails before classification completes, THEN THE Claim_Gate SHALL leave every existing Run_Record, Aggregate_Record, and publication artifact unchanged.

### Requirement 16: Verify Behavior with Layered Testing

**User Story:** As a software maintainer, I want automated checks at unit, property, integration, regression, smoke, and audit levels, so that protocol and evidence invariants remain enforced during implementation.

#### Acceptance Criteria

1. WHEN Dataset_Catalog tests run, THE Testing_Suite SHALL verify Sample_ID uniqueness.
2. WHEN Dataset_Catalog tests run, THE Testing_Suite SHALL verify grouped-task Group_ID completeness.
3. WHEN Dataset_Catalog tests run, THE Testing_Suite SHALL verify Canonical_Order stability under input permutation.
4. WHEN Dataset_Catalog tests run, THE Testing_Suite SHALL verify label-schema conformance.
5. WHEN Dataset_Catalog tests run, THE Testing_Suite SHALL verify exact sample counts.
6. WHEN Dataset_Catalog tests run, THE Testing_Suite SHALL verify Dataset_Hash determinism.
7. WHEN Protocol_Planner tests run, THE Testing_Suite SHALL verify Outer_Train_Set and Outer_Test_Set disjointness.
8. WHEN Protocol_Planner tests run, THE Testing_Suite SHALL verify grouped train and test Group_ID disjointness.
9. WHEN Protocol_Planner tests run, THE Testing_Suite SHALL verify equality between outer-fold coverage and the expected Sample_ID set.
10. WHEN Protocol_Planner tests run, THE Testing_Suite SHALL verify Inner_Fold containment within Outer_Train_Set.
11. WHEN Protocol_Planner tests run, THE Testing_Suite SHALL verify Protocol_Hash round-trip stability.
12. WHEN Preprocessing_Manager tests run, THE Testing_Suite SHALL verify HyperLeaf train-only target normalization.
13. WHEN Preprocessing_Manager tests run, THE Testing_Suite SHALL verify PCA Fit_Sample_IDs Lineage.
14. WHEN Preprocessing_Manager tests run, THE Testing_Suite SHALL verify cache Lineage rejection before cached data is returned.
15. WHEN Preprocessing_Manager tests run, THE Testing_Suite SHALL verify leakage blocking before artifact application.
16. WHEN Model_Registry tests run, THE Testing_Suite SHALL verify Model_Key-to-class identity.
17. WHEN Model_Registry tests run, THE Testing_Suite SHALL verify Input_Contract modality, shape, dtype, order, and preprocessing-key equality.
18. WHEN Model_Registry tests run, THE Testing_Suite SHALL verify proxy labeling.
19. WHEN Model_Registry tests run, THE Testing_Suite SHALL verify Official_Model Provenance requirements.
20. WHEN Model_Registry tests run, THE Testing_Suite SHALL verify the historical list-selector mismatch regression case.
21. WHEN Metric_Engine regression tests run, THE Testing_Suite SHALL compare regression metrics with independently calculated examples using absolute tolerance `1e-12`.
22. WHEN Metric_Engine classification tests run, THE Testing_Suite SHALL compare classification metrics with independently calculated examples using absolute tolerance `1e-12`.
23. WHEN Metric_Engine remote-sensing tests run, THE Testing_Suite SHALL compare remote-sensing metrics with independently calculated examples using absolute tolerance `1e-12`.
24. WHEN Metric_Engine AUROC-tie tests run, THE Testing_Suite SHALL compare results with independently calculated examples using absolute tolerance `1e-12`.
25. WHEN Metric_Engine ECE-boundary tests run, THE Testing_Suite SHALL compare results with independently calculated examples using absolute tolerance `1e-12`.
26. WHEN Metric_Engine zero-variance R² tests run, THE Testing_Suite SHALL verify a null value and reason `zero_variance_target`.
27. WHEN Metric_Engine single-class-fold tests run, THE Testing_Suite SHALL verify a null value and reason `single_class_fold`.
28. WHEN property-based split tests run, THE Testing_Suite SHALL verify group isolation for at least 100 generated grouped Sample_Manifest cases.
29. WHEN property-based split tests run, THE Testing_Suite SHALL verify exact outer coverage for at least 100 generated grouped Sample_Manifest cases.
30. WHEN property-based split tests run, THE Testing_Suite SHALL verify Inner_Fold containment for at least 100 generated grouped Sample_Manifest cases.
31. WHEN property-based split tests run, THE Testing_Suite SHALL verify Canonical_Order invariance for at least 100 generated grouped Sample_Manifest cases.
32. WHEN property-based prediction tests run, THE Testing_Suite SHALL verify merge-order metric invariance using the Numeric_Tolerance for at least 100 generated Prediction_Artifact shard cases.
33. WHEN property-based prediction tests run, THE Testing_Suite SHALL verify duplicate Prediction_Key rejection for at least 100 generated Prediction_Artifact shard cases.
34. WHEN property-based prediction tests run, THE Testing_Suite SHALL verify metric reproducibility using the Numeric_Tolerance for at least 100 generated Prediction_Artifact shard cases.
35. WHEN property-based prediction tests run, THE Testing_Suite SHALL verify exact aggregation membership for at least 100 generated Prediction_Artifact shard cases.
36. WHEN property-based claim tests run, THE Testing_Suite SHALL verify that a Proxy_Model cannot receive `primary_strict` Claim_Status for at least 100 generated Model_Identity and Provenance cases.
37. WHEN synthetic HyperLeaf integration tests run, THE Testing_Suite SHALL execute official-like internal selection without exposing synthetic official-test labels before Selection_Decision freeze.
38. WHEN synthetic HyperLeaf integration tests run, THE Testing_Suite SHALL execute full-training refit after Selection_Decision freeze.
39. WHEN synthetic Histology integration tests run, THE Testing_Suite SHALL execute nested patient selection without exposing outer patient labels before Selection_Decision freeze.
40. WHEN synthetic Histology integration tests run, THE Testing_Suite SHALL execute fixed-epoch LOPO with the epoch policy frozen before held-out patient evaluation.
41. WHEN legacy-import integration tests run, THE Testing_Suite SHALL verify byte-for-byte equality of source values before and after import.
42. WHEN legacy-import integration tests run, THE Testing_Suite SHALL verify that known caveats survive JSON generation.
43. WHEN legacy-import integration tests run, THE Testing_Suite SHALL verify that known caveats survive CSV generation.
44. WHEN legacy-import integration tests run, THE Testing_Suite SHALL verify that known caveats survive Markdown generation.
45. WHEN reporting integration tests run, THE Testing_Suite SHALL compare four-decimal JSON, CSV, and Markdown cells produced from one Aggregate_Record.
46. WHEN audit regression tests run, THE Testing_Suite SHALL verify that an Outer_Test_Set metric cannot invoke checkpoint selection.
47. WHEN audit regression tests run, THE Testing_Suite SHALL verify that validation target normalization cannot read validation targets.
48. WHEN a validation-failure test runs, THE Testing_Suite SHALL verify equality between registry state before and after the rejected operation.
49. WHEN a validation-failure test runs, THE Testing_Suite SHALL verify equality between completed artifact hashes before and after the rejected operation.
50. WHEN a completed-artifact overwrite test runs, THE Testing_Suite SHALL verify rejection before the completed artifact is modified.

### Requirement 17: Capture Reproducibility and Publication Artifacts

**User Story:** As an independent researcher, I want complete run artifacts and environment records, so that strict results can be reconstructed and audited.

#### Acceptance Criteria

1. WHEN a formal run completes, THE Run_Registry SHALL store the Dataset_Hash.
2. WHEN a formal run completes, THE Run_Registry SHALL store the Protocol_Hash.
3. WHEN a formal run completes, THE Run_Registry SHALL store the Model_Identity.
4. WHEN a formal run completes, THE Run_Registry SHALL store the Input_Contract.
5. WHEN a formal run completes, THE Run_Registry SHALL store the configuration hash.
6. WHEN a formal run completes, THE Run_Registry SHALL store the code revision.
7. WHEN a formal run completes, THE Run_Registry SHALL store the Environment_Lock hash.
8. WHEN a formal run completes, THE Run_Registry SHALL store the seed.
9. WHEN a formal run completes, THE Run_Registry SHALL store every transform artifact reference.
10. WHEN a formal run completes, THE Run_Registry SHALL store the checkpoint reference.
11. WHEN a formal run completes, THE Run_Registry SHALL store the Prediction_Artifact reference.
12. WHEN a formal run completes, THE Run_Registry SHALL store the metrics.
13. WHEN a formal run completes, THE Run_Registry SHALL store start and completion timestamps.
14. WHEN a formal run completes, THE Run_Registry SHALL store the Claim_Status.
15. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable Python version.
16. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable PyTorch version.
17. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable NumPy version.
18. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable pandas version.
19. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable Pillow version.
20. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable tifffile version.
21. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable scikit-learn version.
22. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable SciPy version.
23. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable CUDA version.
24. WHEN an Environment_Lock is created, THE Benchmark_System SHALL record the applicable device version or device identity.
25. WHEN an external implementation is used, THE Run_Registry SHALL store the repository URL.
26. WHEN an external implementation is used, THE Run_Registry SHALL store the commit.
27. WHEN an external implementation is used, THE Run_Registry SHALL store the license reference.
28. WHEN an external implementation is used, THE Run_Registry SHALL store the download timestamp.
29. WHEN an external implementation is used, THE Run_Registry SHALL store the source hash.
30. WHEN an external implementation is used, THE Run_Registry SHALL store the local patch hash.
31. WHEN an external implementation is used, THE Run_Registry SHALL store the dependency Environment_Lock.
32. WHEN an external implementation is used, THE Run_Registry SHALL store the upstream reproduction result.
33. WHEN an external implementation is used, THE Run_Registry SHALL store the adaptation note.
34. WHEN a strict result is reproduced from frozen artifacts, THE Metric_Engine SHALL recover every reported point metric with absolute error no greater than `1e-6`.
35. IF a formal run lacks a required artifact, THEN THE Claim_Gate SHALL block `primary_strict` Claim_Status.
36. IF a formal run lacks a required artifact, THEN THE Claim_Gate SHALL identify the missing artifact field in a machine-readable reason.
37. IF a formal run lacks a required hash, THEN THE Claim_Gate SHALL block `primary_strict` Claim_Status.
38. IF a formal run lacks a required hash, THEN THE Claim_Gate SHALL identify the missing hash field in a machine-readable reason.
39. WHEN publication artifacts are exported, THE Report_Generator SHALL use dataset keys instead of local absolute data paths.
40. WHEN publication artifacts are exported, THE Report_Generator SHALL use relative artifact references instead of local absolute artifact paths.
41. WHEN Histology publication artifacts are exported, THE Report_Generator SHALL exclude raw cubes.
42. WHEN Histology publication artifacts are exported, THE Report_Generator SHALL exclude identifying patient mappings.
43. WHEN Histology publication artifacts are exported, THE Report_Generator SHALL exclude unapproved labels.
44. WHEN logs are persisted, THE Benchmark_System SHALL exclude raw sample content.
45. WHEN logs are persisted, THE Benchmark_System SHALL exclude access tokens.
46. WHEN logs are persisted, THE Benchmark_System SHALL exclude secrets.
47. WHEN logs are persisted, THE Benchmark_System SHALL exclude complete environment-variable dumps.
48. WHEN an external checkpoint is registered, THE Benchmark_System SHALL require a source reference before controlled loading.
49. WHEN an external checkpoint is registered, THE Benchmark_System SHALL require a matching content hash before controlled loading.
50. IF an external checkpoint source reference is absent, THEN THE Benchmark_System SHALL block loading before deserialization.
51. IF an external checkpoint content hash is absent or mismatched, THEN THE Benchmark_System SHALL block loading before deserialization.
52. IF external checkpoint validation fails, THEN THE Benchmark_System SHALL emit an Error_Record containing the checkpoint reference, failing field, expected hash when available, and observed hash when available.
53. IF external checkpoint validation fails, THEN THE Benchmark_System SHALL leave the checkpoint file and every completed artifact unchanged.
