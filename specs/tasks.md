# Implementation Plan: CVPR 2027 Hyperspectral Benchmark

## Overview

本计划以 Python 实现设计文档中的统一 benchmark，并保持现有 `src/`、`runs/`、checkpoint 和根目录历史结果不变。实现按基础不变量、数据与协议、训练与预测、registry 与报告、严格任务协议、模型适配和最终自动化执行逐层推进。所有正式比较都从冻结 manifest 和逐样本 prediction artifact 出发。

Windows/PowerShell、现有 TCFM 环境和 RTX 5080 是执行约束。CLI、测试和 benchmark driver 必须支持带空格路径、Windows 原子替换语义、显式 `--device cuda`、可记录的 CUDA/precision 设置和有限资源预算。任务中不得硬编码本机绝对数据路径，publication artifact 只能使用 dataset key 和相对 artifact reference。

剩余工作采用训练优先 remediation lane：既有 audit、adapter、smoke、baseline、screening 和 strict execution 保留为历史记录；新增工作先修复 HyperLeaf outer-variety selection leakage 和 Histology raw-feature lineage，再完成 inner-only optimization 与 corrected strict rerun。只有 corrected HyperLeaf promotion checkpoint 通过后才允许一次 HyperLeaf official 1590→820 评估，official labels 与 `solution.csv` 在此之前保持 sealed。非关键 reporting、official/proxy 扩展、资源列和剩余 PBT 继续后置，且不得以重新运行 smoke 作为 remediation 或 official one-shot 的前置条件。`G:\Datasets\HyperCross` 是唯一工作区，所有 F 盘引用一律忽略；`src/`、`runs/`、旧 checkpoint、现有 float32 templates 和 caches 只读，新输出只写入工作区相对的 `runs_v2/` 与 `artifacts/`。

## Tasks

- [x] 1. 建立 Python benchmark 包、核心 schema 和失败原子性基础
  - [x] 1.1 创建包结构与开发配置
    - 创建 `benchmark/`、`configs/`、`artifacts/`、`runs_v2/` 和 `tests/` 目标结构，并添加 Python package 初始化、typed schema 依赖、pytest 与 Hypothesis 配置。
    - 为 Windows/PowerShell 提供无 Bash 假设的模块入口，保留现有 `src/` 和历史 artifact 为只读输入。
    - _Requirements: 13.33, 13.34, 14.1, 16.1, 16.28, 17.39, 17.40_

  - [x] 1.2 实现 schema-bound 基础类型和验证结果
    - 在 `benchmark/schema/` 定义 IDs、hash、timestamp、enum、validation report、`ErrorRecord`、`ResourceBudget` 和 shared artifact reference。
    - 拒绝未知字段、NaN、Infinity、非法 enum 和非正整数 budget，并输出 machine-readable validation details。
    - _Requirements: 2.10-2.19, 3.10-3.17, 14.1-14.6, 15.37, 17.52_

  - [x] 1.3 实现 canonical codec 和内容哈希
    - 实现 Unicode NFC、UTF-8、字段排序、canonical set/map、ID 规范化、typed binary64 和 UTC timestamp 编码。
    - 提供稳定序列化、对象 hash 和文件流式 hash API；publication 四位小数不得进入科学 artifact hash。
    - _Requirements: 2.6-2.9, 3.19-3.21, 4.29, 11.10-11.12_

  - [x] 1.4 实现 Error Journal 和失败原子 artifact store
    - 实现临时文件写入、flush/hash 校验、同文件系统原子发布、completed-target 预检查和 append-only error journal。
    - 针对 Windows `Path.replace`/共享冲突提供明确错误码；失败只能清理或隔离本次临时文件，不得改动已完成 artifact。
    - _Requirements: 1.27-1.29, 2.20, 3.18, 12.23, 13.3-13.8, 13.35, 13.36, 16.48-16.50_

  - [x] 1.5 实现 Environment Lock、配置冻结和资源预算 schema
    - 采集 Python、PyTorch、NumPy、pandas、Pillow、tifffile、scikit-learn、SciPy、CUDA 和 RTX 5080 device identity，并计算 lock hash。
    - 区分 Hard/Advisory limits，记录 deterministic/performance mode、precision、input volume 和 checkpoint retention policy。
    - _Requirements: 13.2, 13.15, 14.1-14.18, 14.29-14.33, 17.15-17.24_

  - [x]* 1.6 编写 canonical codec、schema、预算和 atomic store 单元测试
    - 覆盖 ID 规范化、集合重排、浮点边界、非法 budget、completed overwrite、Windows 路径与注入式失败。
    - _Requirements: 14.2-14.6, 16.3, 16.6, 16.48-16.50_

  - [x]* 1.7 编写属性测试：Property 30 原子写入与中断安全
    - **Property 30: Atomic write and interruption safety**
    - 生成写入阶段故障，验证临时写入、hash 后发布、completion exactly once 和旧 artifact 不变。
    - **Validates: Requirements 13.3-13.8, 13.35, 13.36**

  - [x]* 1.8 编写属性测试：Property 34 资源预算有效性与执行规则
    - **Property 34: Resource budget validity and enforcement**
    - 生成合法/非法 limits 和运行测量，验证调度前拒绝、hard checkpoint stop、advisory warning 与完整测量记录。
    - **Validates: Requirements 14.1-14.18, 14.29-14.33**

  - [x]* 1.9 编写属性测试：Property 38 失败原子性与错误记录完整性
    - **Property 38: Failure atomicity and Error Record completeness**
    - 对验证边界注入失败，比较前后 registry snapshot 与 completed hashes，并检查 error stage、IDs、field、expected/observed 和 invariants。
    - **Validates: Requirements 1.28, 1.29, 2.11, 2.13, 2.15, 2.17, 2.19, 3.11, 3.13, 3.15, 3.17, 3.18, 4.12, 4.13, 4.27, 4.28, 5.17, 5.18, 6.29, 6.30, 16.48-16.50**

- [x] 2. 实现 Dataset Catalog 和不可变 Sample Manifest
  - [x] 2.1 实现 dataset spec、sample record 和 manifest schema
    - 定义 dataset key、task kind、expected count、label schema、group requirement、输入引用/hash 和 canonical sample order。
    - 为 HyperLeaf、Histology 和 remote sanity track 提供 adapter protocol，不在基础层硬编码本机路径。
    - _Requirements: 2.1-2.6, 17.39_

  - [x] 2.2 实现 manifest materialization、验证和版本发布
    - 流式读取样本元数据并验证 exact count、唯一 ID、Group_ID、label dimension 和必需输入可读性。
    - 内容或标签变化必须产生新 `Dataset_Hash` 和新不可变 manifest；失败不得允许 protocol creation。
    - _Requirements: 2.1-2.23_

  - [x] 2.3 实现 Histology 稳定脱敏导出
    - 使用稳定、不可逆的 patient-to-Group_ID 映射生成公开 manifest，并从公开输出移除原 patient identifier、raw cube 与未批准 labels。
    - _Requirements: 2.24, 2.25, 17.41-17.43_

  - [x]* 2.4 编写 Dataset Catalog 单元测试
    - 覆盖 Sample_ID 唯一性、Group_ID 完整性、label schema、exact count、unreadable input 和 manifest immutable versioning。
    - _Requirements: 2.10-2.23, 16.1-16.6_

  - [x]* 2.5 编写属性测试：Property 1 canonical dataset identity
    - **Property 1: Canonical dataset identity**
    - 至少生成 100 组输入顺序 permutation，并验证相同内容的 order/serialization/hash 相同及内容变化后旧 bytes 不变。
    - **Validates: Requirements 2.6-2.9, 2.21-2.23**

  - [x]* 2.6 编写属性测试：Property 2 manifest 完整性与失败原子性
    - **Property 2: Manifest completeness and failure atomicity**
    - 生成 duplicate、missing group、bad labels、count mismatch 和 unreadable inputs，验证 protocol 前阻断及旧 artifact 不变。
    - **Validates: Requirements 2.1-2.5, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20**

  - [x]* 2.7 编写属性测试：Property 3 Histology 稳定脱敏
    - **Property 3: Stable Histology de-identification**
    - 验证重复导出 mapping 相同、原 patient identifiers 不出现且受限数据不进入公开 artifact。
    - **Validates: Requirements 2.24, 2.25, 17.41-17.43**

- [x] 3. 实现 protocol-first Planner 和可重用 fold manifests
  - [x] 3.1 实现 protocol/fold/selection-policy/evaluation-budget schema
    - 定义 outer/inner folds、canonical sample sets、Seed_Set、Protocol_Key、Protocol_Hash 和 successful-evaluation budget。
    - _Requirements: 3.1, 3.2, 5.21-5.24, 8.1-8.3, 9.17-9.19_

  - [x] 3.2 实现通用 split planner 与不变量验证器
    - 验证 outer sample/group isolation、exact coverage、inner containment 和 inner group isolation，并在模型构建前原子发布 protocol。
    - _Requirements: 3.1-3.18_

  - [x] 3.3 实现 protocol round-trip、comparability 和 dry-run API
    - 支持序列化/重载保持 membership/hash；比较模型时强制相同 protocol hash 与 fold sample sets。
    - 提供只规划不构建模型的 Python/PowerShell-safe dry-run 入口。
    - _Requirements: 3.8, 3.9, 3.19-3.21, 6.31_

  - [x]* 3.4 编写 Protocol Planner 单元测试
    - 覆盖 sample/group disjointness、coverage、containment、round-trip 和模型构建调用顺序。
    - _Requirements: 3.1-3.21, 16.7-16.11_

  - [x]* 3.5 编写属性测试：Property 4 protocol-before-model 和 canonical round trip
    - **Property 4: Protocol-before-model and canonical round trip**
    - 至少生成 100 个等价 fold 顺序，验证 model build 晚于 protocol publication 且 membership/hash 往返稳定。
    - **Validates: Requirements 3.1, 3.2, 3.19-3.21**

  - [x]* 3.6 编写属性测试：Property 5 outer sample/group isolation
    - **Property 5: Outer sample and group isolation**
    - 至少生成 100 个 grouped manifests，验证 outer train/test 不相交、exact coverage 和 group isolation。
    - **Validates: Requirements 3.3-3.5, 3.10, 3.12, 3.14**

  - [x]* 3.7 编写属性测试：Property 6 nested containment
    - **Property 6: Nested containment**
    - 至少生成 100 个 nested grouped folds，验证 inner disjointness、outer-train containment 和 group boundary。
    - **Validates: Requirements 3.6, 3.7, 3.16, 8.17, 8.18, 9.2, 9.4-9.6**

- [~] 4. Checkpoint - 验证基础、manifest 和 protocol 层
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. 实现 fold-scoped preprocessing、lineage cache 和高效 cube 访问
  - [x] 5.1 实现通用 Fitted Artifact 和 Preprocessing Manager
    - 实现 target normalization、PCA、band normalization、imputation、class weights、threshold 和 calibration 的 fit/apply API。
    - 每个 artifact 保存精确 Fit_Sample_IDs，并在 apply 前验证 inner-train 或 outer-train 边界。
    - _Requirements: 4.1-4.16, 9.33, 9.34_

  - [x] 5.2 实现 PCA identity 和 Feature Cache
    - cache key 绑定 dataset、protocol、fold、representation 和 transform-state hash；lineage 保存 source/config/fit/transform hashes 与 output shape。
    - 所有 mismatch 必须在返回 cached tensor 前拒绝，且不得改动 rejected entry。
    - _Requirements: 4.17-4.28, 14.21-14.28_

  - [x] 5.3 实现 chunked 和 memory-mapped hyperspectral reader
    - 为 TIFF/ENVI raw cube 提供 shape/dtype 检查、chunked bands/spatial windows 和只读 memmap；避免默认物化完整 Histology cube。
    - _Requirements: 14.19, 14.20, 17.41, 17.44_

  - [x]* 5.4 编写 preprocessing、cache 和 raw reader 单元测试
    - 覆盖 train-only target statistics、PCA fit IDs、cache rejection、shape mismatch、deterministic state hash 和 memmap/chunk boundaries。
    - _Requirements: 4.1-4.29, 14.19-14.28, 16.12-16.15, 16.47_

  - [x]* 5.5 编写属性测试：Property 7 fold-scoped fitted state
    - **Property 7: Fold-scoped fitted state**
    - 随机污染 fit IDs，验证 inner/outer boundary 和 Outer_Test_Set leakage 在 apply 前阻断。
    - **Validates: Requirements 4.1-4.12**

  - [x]* 5.6 编写属性测试：Property 8 HyperLeaf target normalization isolation
    - **Property 8: HyperLeaf target normalization isolation**
    - 验证统计量仅来自许可 training IDs，不读取 validation 或 official-test targets。
    - **Validates: Requirements 4.14-4.16, 16.12, 16.47**

  - [x]* 5.7 编写属性测试：Property 9 PCA 和 cache lineage
    - **Property 9: PCA and cache lineage**
    - 生成 source、transform、fit boundary、shape 和 key mismatch，验证返回前拒绝与完整 Error_Record。
    - **Validates: Requirements 4.17-4.27, 14.21-14.27**

  - [x]* 5.8 编写属性测试：Property 10 transform determinism
    - **Property 10: Transform determinism**
    - 验证相同 inputs/config/seed 的 fitted state hash 一致。
    - **Validates: Requirements 4.29**

  - [x]* 5.9 编写属性测试：Property 35 高效 raw access 与 immutable cache reuse
    - **Property 35: Efficient raw access and immutable cache reuse**
    - 验证 chunk/memmap 结果等价、source hash mismatch 拒绝并保持旧 cache bytes。
    - **Validates: Requirements 14.19, 14.20, 14.26-14.28**

- [x] 6. 实现稳定 Model Registry、Input Contract 和 Provenance
  - [x] 6.1 实现 registry resolution 和 model identity validation
    - 一个 Model_Key 只解析到一个 identity/contract；build 后核对类全限定名、implementation kind、参数量和 source revision。
    - CLI 必须只按稳定 Model_Key 选择模型，禁止位置索引或易漂移 list selector。
    - _Requirements: 6.1-6.8, 6.24-6.31_

  - [x] 6.2 实现 Input Contract 验证
    - 在训练前比较 modalities、tensor shapes、dtypes、channel/band order、Representation_Key 和 preprocessing key。
    - _Requirements: 6.2, 6.7-6.13, 6.28-6.30_

  - [x] 6.3 实现 official/adapted/proxy provenance 和 Reproduction Gate
    - 保存 upstream URL/commit/license/source hash/local patch hash/reproduction status/adaptation/ignored modalities。
    - 强制 proxy display name 含 `proxy`；没有通过 upstream reproduction record 时阻断 formal adaptation。
    - _Requirements: 6.14-6.23, 7.21-7.24, 17.25-17.33_

  - [x] 6.4 实现 external checkpoint 受控加载器
    - 在反序列化前验证 source reference 和 content hash，使用受限加载 API，并在失败时保持 checkpoint bytes 与 completed artifacts 不变。
    - _Requirements: 17.48-17.53_

  - [x]* 6.5 编写 Model Registry、Provenance 和 checkpoint 单元测试
    - 覆盖 key/class/kind mismatch、完整 contract、proxy label、official provenance、历史 selector mismatch 和 pre-deserialization hash gate。
    - _Requirements: 6.1-6.31, 7.21-7.24, 16.16-16.20, 17.48-17.53_

  - [x]* 6.6 编写属性测试：Property 13 model identity 和 input contract
    - **Property 13: Stable model identity and input contract**
    - 随机改变 class/kind/provenance/contract 字段，验证训练前失败和 completed state 不变。
    - **Validates: Requirements 6.1-6.13, 6.24-6.31**

  - [x]* 6.7 编写属性测试：Property 14 official/proxy provenance safety
    - **Property 14: Official and proxy provenance safety**
    - 验证 official/adapted 字段完整，且任意 proxy metadata 都不能超过 `proxy_only`。
    - **Validates: Requirements 6.14-6.23, 7.21-7.24**

- [x] 7. 实现 append-only Run Registry、Aggregate storage 和 Recovery Manager
  - [x] 7.1 实现 Run/Artifact/Aggregate append-only registry
    - 持久化 Frozen_Run_Identity、run status、artifact references、metrics、timestamps、claim 和 provenance；禁止原地更新 completed record。
    - _Requirements: 13.1, 13.2, 13.31-13.36, 17.1-17.14_

  - [x] 7.2 实现 supersession 和 canonical completed selection
    - 修正结论时追加新 record 并引用旧 ID；重复 COMPLETE jobs 按 completed_utc、Run_ID 规则选至多一个 canonical member，保留全部 duplicates。
    - _Requirements: 1.25, 1.26, 12.17, 12.18, 13.31, 13.32_

  - [x] 7.3 实现 recovery artifacts、resume validation 和 missing-job scheduler
    - checkpoint boundary 保存 model/optimizer/scheduler/stage；恢复前比较 dataset/protocol/model/config/fold/seed/environment。
    - missing jobs 精确计算 declared Cartesian product 与 canonical completed jobs 的差集。
    - _Requirements: 13.6-13.32_

  - [x]* 7.4 编写 registry/recovery 单元测试
    - 覆盖 interrupted statuses、last stage、resume mismatch、新 Run_ID 要求、duplicate audit 和 exact missing jobs。
    - _Requirements: 13.1-13.36, 16.48-16.50_

  - [x]* 7.5 编写属性测试：Property 29 frozen-identity resume safety
    - **Property 29: Frozen-identity resume safety**
    - 随机改变 frozen fields，验证只在全兼容时恢复四类状态，任一 mismatch 不恢复且要求新 Run_ID。
    - **Validates: Requirements 13.1, 13.2, 13.9-13.29**

  - [x]* 7.6 编写属性测试：Property 31 exact missing jobs 与 canonical duplicate selection
    - **Property 31: Exact missing jobs and canonical duplicate selection**
    - 生成 declared/completed job sets 和 duplicates，验证差集精确且 canonical member 唯一。
    - **Validates: Requirements 13.30-13.32**

  - [x]* 7.7 编写属性测试：Property 39 完整 reproducibility record
    - **Property 39: Complete reproducibility record**
    - 生成 formal run 与 external provenance，验证全部 hash、artifact、environment、timestamp 和 claim 字段完整。
    - **Validates: Requirements 17.1-17.33, 17.35-17.38**

- [x] 8. Checkpoint - 验证 preprocessing、model identity 和 recovery 层
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. 实现 Training Engine、Selection Controller 和 one-shot Outer Evaluator
  - [x] 9.1 实现 inner training 和 checkpoint lifecycle
    - 使用冻结 config/seed 执行 inner training，记录 epochs、steps、runtime、peak memory、device、precision、parameter count 和 input volume。
    - 支持 selected/last/recovery checkpoint retention 及 Hard_Limit checkpoint-boundary stop。
    - _Requirements: 13.16-13.19, 14.7-14.18, 14.29-14.33_

  - [x] 9.2 实现 Selection Decision 冻结
    - 保存 candidate set/scores/source Sample_IDs/metric direction/tie-break/source prediction hash，并只接受 Outer_Train_Set 内 inner evidence。
    - refit 必须把未改变的 frozen decision 应用于完整 Outer_Train_Set。
    - _Requirements: 5.1-5.18, 8.4-8.8, 9.5-9.8, 10.1-10.11_

  - [x] 9.3 实现 outer-label capability gate 和 evaluation lease
    - selector、fitter 和 trainer 不具 outer-label access；只有 frozen checkpoint evaluator 可 reserve budget、禁用 gradient/fitted-state update 并成功后 commit 一次。
    - _Requirements: 5.13, 5.15-5.24, 8.6-8.8, 9.7, 9.8_

  - [x] 9.4 实现统一训练编排 API 和可恢复 stage machine
    - 连接 protocol、preprocessing、registry、training、selection、refit 和 evaluator；任何异常追加 FAILED/ABORTED record 与 Error_Record。
    - _Requirements: 3.1, 4.8-4.13, 5.13-5.24, 13.1-13.19_

  - [x]* 9.5 编写 Training/Selection/Outer Evaluator 单元测试
    - 使用 fake label vault 和 toy models 验证 outer labels 在 freeze 前不可访问、budget 只成功消费、超预算不加载 labels。
    - _Requirements: 5.1-5.24, 16.37-16.40, 16.46_

  - [x]* 9.6 编写属性测试：Property 11 no outer tuning
    - **Property 11: No outer tuning**
    - 生成 selection spaces 和 contaminated source IDs，验证 decision 完整性、outer exclusion 和 freeze-before-access。
    - **Validates: Requirements 5.1-5.17**

  - [x]* 9.7 编写属性测试：Property 12 one-shot evaluation budget
    - **Property 12: One-shot evaluation budget**
    - 生成成功、失败、重试和超预算序列，验证 successful count exactly once 且 completed predictions 不变。
    - **Validates: Requirements 5.19-5.24, 8.6, 8.7, 9.7, 9.8**

- [x] 10. 实现 prediction-first metrics、statistics 和 aggregate builder
  - [x] 10.1 实现 Prediction Artifact schema、分片写入和完整性校验
    - 逐样本保存 `(Sample_ID, Fold_ID, seed, Inference_Branch)`、truth、prediction/probability；验证 expected coverage 和 key uniqueness 后原子发布。
    - _Requirements: 11.1-11.9_

  - [x] 10.2 实现回归、二分类和 remote-sensing metrics
    - 实现每目标 R²/mean R²/MAE、accuracy/balanced accuracy/AUROC/sensitivity/specificity/confusion matrix/ECE、OA/AA/Kappa。
    - single-class AUROC 与 zero-variance R² 返回 null 和固定 reason；metric schema 声明 Numeric_Tolerance。
    - _Requirements: 8.9-8.13, 9.9-9.26, 11.9-11.20, 16.21-16.27_

  - [x] 10.3 实现 canonical shard merge 和 Aggregate Builder
    - 只加载明确 Member_Run_IDs 的完整预测，按 canonical prediction key 合并，保存每个 prediction hash、seed/fold variance 和 exact membership。
    - _Requirements: 11.11, 11.12, 11.17-11.23_

  - [x] 10.4 实现 patient bootstrap 和 paired comparison
    - 以完整 patient Group_ID 为重采样单位，固定 10,000 次、双侧 95%、预声明 seed；paired comparison 仅重采样共同 patient IDs。
    - _Requirements: 9.27-9.32, 11.24-11.31_

  - [x]* 10.5 编写 metric engine 独立 oracle 单元测试
    - 用手算 fixtures 以 `1e-12` 验证 regression/classification/remote metrics、AUROC ties、ECE boundaries 和 undefined reasons。
    - _Requirements: 16.21-16.27_

  - [x]* 10.6 编写属性测试：Property 18 patient-level uncertainty
    - **Property 18: Patient-level uncertainty**
    - 验证 10,000 次 patient resamples、双侧 95%、seed 记录和 candidate/control 配对保持。
    - **Validates: Requirements 9.27-9.32, 11.24-11.31**

  - [x]* 10.7 编写属性测试：Property 22 prediction key uniqueness 与 exact coverage
    - **Property 22: Prediction key uniqueness and exact coverage**
    - 至少生成 100 个 prediction shard cases，验证 missing/unexpected/duplicate key 在 metrics 前拒绝。
    - **Validates: Requirements 11.1-11.8**

  - [x]* 10.8 编写属性测试：Property 23 prediction-first metric reproducibility
    - **Property 23: Prediction-first metric reproducibility**
    - 至少生成 100 个 prediction/config cases，验证只接受 validated artifacts、重复计算符合 tolerance，strict 重算符合 `1e-6`。
    - **Validates: Requirements 11.9, 11.10, 17.34**

  - [x]* 10.9 编写属性测试：Property 24 merge-order invariance 与 undefined metrics
    - **Property 24: Merge-order invariance and undefined metrics**
    - 至少生成 100 个 shard permutations，并覆盖 single-class/zero-variance 边界。
    - **Validates: Requirements 11.11-11.16**

  - [x]* 10.10 编写属性测试：Property 25 named variance 与 exact aggregate membership
    - **Property 25: Named variance and exact aggregate membership**
    - 验证 seed/fold variance 命名、aggregation policy 固定和 member/prediction hash 精确对应。
    - **Validates: Requirements 11.17-11.23**

  - [x]* 10.11 编写 aggregate/statistics 集成测试
    - 覆盖 shard 重排、患者 bootstrap 固定 seed、paired IDs、fold-macro 与 pooled OOF 分离和 incomplete member rejection。
    - _Requirements: 8.20-8.23, 9.9-9.32, 11.11-11.31_

- [x] 11. 实现 Claim Gate、authoritative reporting 和同步发布
  - [x] 11.1 实现固定 precedence 的 Claim Gate
    - 计算全部适用 ceilings，返回唯一 status、supporting/blocking reasons 和 prerequisite checks；leakage/identity mismatch 强制 invalid。
    - _Requirements: 15.1-15.27_

  - [x] 11.2 实现 benchmark tier、comparability 和 promotion gate
    - 按 Classic/In_Repo/Official/Adapted/Proxy 分组；不同 Protocol_Hash 不计算 rank/improvement；不同 representation 要求 matched control。
    - incomplete folds/seeds 不创建 Publication_Aggregate。
    - _Requirements: 7.1-7.20, 10.12-10.21, 12.11-12.16_

  - [x] 11.3 实现单一 Publication Matrix 和三格式 renderer
    - 从 Run Registry/Aggregate 生成一个 matrix，再渲染 JSON/CSV/Markdown；所有显示数值 round-half-even 为四位小数字符串并保留尾随零。
    - _Requirements: 12.1-12.10, 15.32-15.36_

  - [x] 11.4 实现 cross-format validator、hand-edited conflict gate 和 transactional publish
    - 比较三格式 parsed cells、model/protocol/claim；检测手工 summary 冲突时在写 publication files 前拒绝。
    - _Requirements: 12.17-12.23_

  - [x] 11.5 实现安全 publication projection 和审计输出
    - 只输出 dataset keys/relative refs；过滤 raw content、patient mapping、未批准 labels、tokens、secrets 和完整环境变量。
    - 保留 task-level conclusions，不生成跨 HyperLeaf/Histology/remote 的 overall score。
    - _Requirements: 15.28-15.36, 17.39-17.47_

  - [x]* 11.6 编写 Claim Gate 与 reporting 单元测试
    - 覆盖 ceiling precedence、incomplete aggregate、protocol separation、tier columns、FixedDecimal4、supersession 和 conflict rejection。
    - _Requirements: 7.1-7.20, 12.1-12.23, 15.1-15.36_

  - [x]* 11.8 编写属性测试：Property 26 四位小数跨格式一致性
    - **Property 26: Four-decimal cross-format consistency**
    - 生成边界浮点与 model/protocol/claim records，验证 JSON/CSV/Markdown 每个 cell 完全一致。
    - **Validates: Requirements 12.1-12.10**

  - [x]* 11.9 编写属性测试：Property 27 publication aggregate completeness
    - **Property 27: Publication aggregate completeness**
    - 生成 missing folds/seeds 与 supersession/caveat cases，验证 INCOMPLETE 阻断和引用保留。
    - **Validates: Requirements 12.11-12.20**

  - [ ]* 11.7 编写属性测试：Property 15 protocol comparability、tiers 和 matched controls
    - **Property 15: Protocol comparability, model tiers, and matched controls**
    - 仅在 strict multi-seed 与 official one-shot checkpoint 后修复缺失资源列，再随机组合 protocol/representation/model kind，验证可比较性、报告列、control gate 和 remote ranking exclusion。
    - **Validates: Requirements 3.8, 3.9, 7.1-7.20, 10.12-10.17**

  - [~]* 11.10 编写属性测试：Property 28 authoritative report transaction
    - **Property 28: Authoritative report transaction**
    - official one-shot 后注入 summary conflict 和 renderer validation failure，验证 publication transaction 不改旧文件。
    - **Validates: Requirements 12.21-12.23**

  - [~]* 11.11 编写属性测试：Property 36 claim ceiling determinism
    - **Property 36: Claim ceiling determinism**
    - official one-shot 后至少生成 100 个 ceiling/evidence sets，验证结果等于固定 precedence minimum 且高分不提升 ceiling。
    - **Validates: Requirements 15.1-15.27**

  - [~]* 11.12 编写属性测试：Property 37 track-level claim separation
    - **Property 37: Track-level claim separation**
    - official one-shot 后验证三个 track 分别保留 conclusion，headline/abstract fields 完整且无 combined overall score。
    - **Validates: Requirements 15.28-15.38**

  - [~]* 11.13 编写属性测试：Property 40 安全 publication 与 checkpoint loading
    - **Property 40: Safe publication and checkpoint loading**
    - official one-shot 后生成敏感路径/字段和 checkpoint metadata，验证公开输出过滤及 hash/source gate 早于 deserialization。
    - **Validates: Requirements 17.39-17.53**

  - [x] 13.7 审计唯一工作区、TCFM、CUDA 与数据可读性
    - 只解析 `G:\Datasets\HyperCross`、`G:\Datasets\HyperLeaf2024` 和 `G:\Datasets\PKG - HistologyHSI-GB`；发现 F 盘或工作区外写路径时立即阻断，不迁移或跟随该引用。
    - 在 TCFM 中记录 Python/PyTorch/CUDA、RTX 5080 identity、可用显存、关键依赖、数据样本数与首批 shape/dtype；输出写入 `artifacts/audit/` 的相对路径。
    - _Requirements: 2.10, 2.18, 14.1-14.14, 17.15-17.24, 17.39, 17.40_

  - [x] 13.8 验证现有 G 盘 checkpoint 可重载与推理兼容性
    - 对 `runs/` 和 G 盘 checkpoint 只读扫描，按 source hash、模型类、state-dict keys、input contract 和 preprocessing metadata 验证可重载；不得覆盖或修改旧 checkpoint。
    - 使用本地 RTX 5080 与 TCFM 做 bounded 单批次前向验证，输出到 `runs_v2/audit/`，并为不兼容 checkpoint 记录明确 negative evidence 与重新训练原因。
    - _Requirements: 1.1-1.5, 6.3-6.13, 13.33-13.36, 17.48-17.53_

  - [x] 13.9 冻结训练 fast-lane 运行配置与恢复边界
    - 为后续 smoke、baseline、screening、strict 和 official 阶段固定 `--device cuda`、TCFM、RTX 5080 单进程、precision、seed、hard/advisory budget、checkpoint interval 与工作区相对输出根。
    - 每个训练 job 必须写可恢复 checkpoint 和 Frozen_Run_Identity；禁止从 F 盘恢复，禁止原地修改 `src/`、`runs/` 或旧 checkpoint。
    - _Requirements: 13.1-13.29, 14.1-14.18, 14.29-14.33, 17.15-17.24_

  - [x] 13.1 实现只读 legacy source scanner 和 importer
    - 读取现有 JSON/CSV/Markdown、runner metadata 和 checkpoint references，保存 source path/hash、原 protocol 描述、原始数值和 caveats，不重写源文件。
    - validation failure 必须在 Run_ID 分配前阻断。
    - _Requirements: 1.1-1.11, 1.25-1.29, 13.33, 13.34_

  - [~] 13.2 编码已知 legacy model/category/claim 映射
    - 对 `0.9153 ± 0.0686`、固定 9/4、`0.9392` ensemble、lite proxies、Indian Pines、strict candidates 和 selector mismatch 生成确定 ceiling 与解释。
    - _Requirements: 1.12-1.24, 15.5-15.18_

  - [~] 13.3 实现 legacy negative evidence 与 superseding records
    - 把 underperformance、seed instability 和 representation ablation 作为 append-only evidence 导入；新结论引用旧记录但不改旧值。
    - _Requirements: 1.25, 1.26, 10.22-10.25, 12.17-12.20_

  - [ ]* 13.4 编写 legacy import 集成与 byte-preservation 测试
    - 对仓库 fixture 验证导入前后 source bytes/hash 相同，caveats 在 JSON/CSV/Markdown 中保留，非法源不分配 Run_ID。
    - _Requirements: 1.1-1.29, 13.33, 13.34, 16.41-16.44_

  - [ ]* 13.5 编写属性测试：Property 32 legacy immutability 与 supersession
    - **Property 32: Legacy evidence immutability and supersession**
    - 生成 legacy records 与修正，验证 path/hash/protocol/metrics/caveats 原样保存和 append-only supersession。
    - **Validates: Requirements 1.1-1.5, 1.25-1.29, 13.33, 13.34**

  - [ ]* 13.6 编写属性测试：Property 33 legacy claim ceilings
    - **Property 33: Legacy claim ceilings**
    - 生成 missing provenance、outer selection、proxy、sanity、single run 和 selector mismatch，验证确定 ceiling/status。
    - **Validates: Requirements 1.6-1.24**

- [ ] 14. 接入 HyperLeaf fast lane、筛选候选并执行严格与 official 评估
  - [x] 14.1 实现 HyperLeaf dataset adapter 和 representation builders
    - 从 `G:\Datasets\HyperLeaf2024` materialize 1,590 train/820 official-test manifests、wheat variety Group_ID、visible RGB、global spectrum、PCA pseudo-RGB 与 full-cube-compatible keys；运行时配置只保存 dataset key，不把绝对路径写入 publication artifact。
    - target normalization 只能使用许可 training IDs；official labels 在候选冻结前不可访问。
    - _Requirements: 2.1-2.23, 4.14-4.21, 8.1-8.3_

  - [x] 14.4 适配 HyperLeaf classic 与 matched-input controls
    - 将 PLSR、RF 和必要的相同 Representation_Key 控制接入 registry、fold-scoped preprocessing、统一 prediction artifact 与相同 Seed_Set/resource budget。
    - 新 runner 调用现有实现时只通过 adapter，不修改只读 `src/`、`runs/` 或旧 checkpoint。
    - _Requirements: 7.6-7.14, 8.9-8.13, 10.12-10.17_

  - [x] 14.5 适配 HyperLeaf in-repo Route A 模型
    - 按稳定 Model_Key 适配 global fusion、target-specific experts、shared encoder/heads、token/local ablations，并修复历史 selector 漂移路径。
    - Global Route A metadata 必须声明 global fusion baseline，不声称 single-token correspondence；新候选必须由 adapter/新 runner 接入。
    - _Requirements: 1.21, 1.23, 1.24, 6.31, 10.8, 10.9, 10.32, 10.33_

  - [x] 14.9 执行 HyperLeaf RTX 5080 bounded smoke
    - 在 TCFM 和本地 RTX 5080 上使用 tiny train-only subset 与硬预算验证 CUDA、显存、RGB/光谱/PCA shape、finite loss、反向传播、可恢复 checkpoint、重载与逐样本 prediction artifact；不得读取 official test labels。
    - 单进程运行，输出仅写 `runs_v2/smoke/hyperleaf/` 和 `artifacts/` 相对路径，checkpoint boundary 支持 resume。
    - _Requirements: 5.13-5.24, 8.28-8.30, 13.1-13.29, 14.7-14.18, 16.37, 16.38_

  - [x] 14.10 复现 G 盘当前可运行 HyperLeaf baseline
    - 在 TCFM 和本地 RTX 5080 上从只读 G 盘代码/checkpoint 复现当前可运行 Route A、PLSR/RF 及适用 matched controls，先用原协议核对预测与记录的容差，再通过新 runner 生成 baseline prediction artifacts。
    - 单进程运行，输出仅写 `runs_v2/baselines/hyperleaf/`，保存可恢复 checkpoint、环境锁、数据/协议/model hashes；任何差异作为 caveat，不改旧结果。
    - _Requirements: 1.1-1.5, 6.1-6.13, 11.1-11.10, 13.33, 13.34, 17.1-17.24_

  - [x] 14.3 实现 `leaf.lovo.nested.v1` planner 和 runner wiring
    - 每个完整 variety 恰好一次 outer test；route/epoch selection 仅用 outer-train inner varieties，输出 per-fold/per-target/fold-macro/pooled OOF。
    - runner 固定 TCFM、本地 RTX 5080 单进程、工作区相对输出和 fold-boundary 可恢复 checkpoint；不得访问 official test。
    - _Requirements: 8.14-8.23, 8.28-8.30_

  - [x] 14.11 在 G 盘重建缺失的 HyperLeaf 候选 runner
    - 在新 runner 中重建 independent-target、style-augmentation、train-fold PCA 和 RGB/PCA-consistency 分支，并为 Gsw route/loss/transform 暴露预先冻结的候选配置；不得修改 legacy `src/`。
    - 所有训练固定 TCFM、本地 RTX 5080 单进程、`runs_v2/candidates/hyperleaf/` 相对输出与可恢复 checkpoint，并强制 outer-train-only normalization/PCA。
    - _Requirements: 4.14-4.21, 5.1-5.18, 10.1-10.11, 13.1-13.29_

  - [x] 14.12 用 inner varieties 与 LOVO seed 42 筛选 HyperLeaf Gsw/grouped 候选
    - 以 seed 42 和小预算比较 independent-target、style、PCA、RGB/PCA consistency 及 Gsw 专项候选；只使用 inner varieties/LOVO grouped evidence，禁止读取 official test images 的 labels 或用 official metric 做选择。
    - 在 TCFM 和本地 RTX 5080 上单进程运行，输出到 `runs_v2/screening/hyperleaf/`，逐候选保存可恢复 checkpoint/predictions；失败或退化候选 append-only 记录 negative evidence。
    - _Requirements: 5.1-5.18, 8.14-8.23, 10.1-10.11, 10.22-10.25_

  - [x] 14.13 对冻结的 HyperLeaf 胜出候选执行 strict LOVO 多种子
    - 在 candidate-screening checkpoint 通过后冻结候选空间、route、epoch policy、preprocessing 与预算，只对胜出候选和 matched controls 执行 seeds `42/123/7` 的 nested LOVO。
    - 使用 TCFM、本地 RTX 5080 单进程、`runs_v2/strict/hyperleaf/` 相对输出和 fold/epoch-boundary 可恢复 checkpoint；完整输出 per-fold、per-target、fold-macro 与 pooled OOF predictions/metrics。
    - _Requirements: 8.14-8.23, 10.12-10.31, 11.1-11.23, 13.6-13.32_

  - [~] 14.2 实现并执行 `leaf.official.v1` one-shot runner
    - 本任务显式依赖 task 24.6 的 corrected promotion checkpoint；在该 checkpoint 原子发布并通过前保持 blocked，不得把 task 14.12/14.13 的 development-only evidence 用作解封依据。
    - 只有 corrected leakage-safe strict multi-seed 胜出候选、matched controls、最终 full-train config、precision identity 和 epoch policy 完全冻结后，才在 1,590 train 上全量 refit，并通过 one-shot lease 对 820 official samples 评估一次；任何筛选逻辑不得读取 official labels 或 `solution.csv`。
    - 使用 TCFM、本地 RTX 5080 单进程、`runs_v2/official/hyperleaf/` 相对输出和可恢复 refit checkpoint；成功评估后立即提交唯一 prediction artifact 与预算消费记录。不得要求或重跑 smoke validation。
    - _Requirements: 5.19-5.24, 8.1-8.13, 13.1-13.29_

  - [ ]* 14.6 编写 synthetic HyperLeaf strict integration tests
    - 使用小型 grouped fixtures 验证 official-like inner selection、freeze、full-train refit、label vault 和 LOVO exact variety coverage。
    - _Requirements: 8.1-8.30, 16.37, 16.38_

  - [ ]* 14.7 编写属性测试：Property 16 HyperLeaf protocol separation
    - **Property 16: HyperLeaf protocol separation**
    - 验证 official train/test isolation、LOVO variety exact-once coverage 和 official/LOVO/random 分区报告。
    - **Validates: Requirements 8.1-8.30**

  - [ ]* 14.8 编写 HyperLeaf metric/selector 回归测试
    - 验证每目标/mean R²/MAE 命名、random split `development` ceiling、旧 route/token selector mismatch 不可重现。
    - _Requirements: 1.21, 1.23, 8.9-8.13, 8.24-8.27, 16.20, 16.46, 16.47_

- [x] 15. 接入 Histology fast lane、筛选 matched candidates 并执行 strict 评估
  - [x] 15.1 实现 Histology dataset adapter 和受治理 representations
    - 从 `G:\Datasets\PKG - HistologyHSI-GB` materialize ROI Sample_ID、patient Group_ID、labels/input hashes；明确 cached spatial-mean PCA 与 Full_Cube spatial PCA 的不同 Representation_Key。
    - 公开导出使用稳定脱敏 group IDs，raw cube 使用只读 chunk/memmap；运行配置不得跟随任何 F 盘引用。
    - _Requirements: 2.3-2.5, 2.24, 2.25, 9.37, 9.38, 14.19, 14.20, 17.41-17.43_

  - [x] 15.5 适配 Histology Local Route A、MIL-concat 和 matched controls
    - 使用稳定 keys 接入 PCA-only、RGB-only、Local Route A、MIL-concat、attention-disabled control、classic SVM/RF 和固定 inference branch。
    - 新候选先通过 adapter/新 runner 接入，不修改只读 `src/`、`runs/` 或旧 checkpoint；缺 attention-specific control 时只报告 combined method。
    - _Requirements: 7.6-7.14, 10.10-10.17, 10.34_

  - [x] 15.10 执行 Histology RTX 5080 bounded smoke
    - 在 TCFM 和本地 RTX 5080 上使用少量 patient-grouped ROI 与硬预算验证 CUDA、显存、RGB/PCA/local-token shape、finite loss、反向传播、可恢复 checkpoint、重载和逐样本 prediction artifact。
    - 单进程运行，输出仅写 `runs_v2/smoke/histology/` 和 `artifacts/` 相对路径；smoke 不消费 formal outer evaluation budget。
    - _Requirements: 5.13-5.24, 9.39-9.41, 13.1-13.29, 14.7-14.20, 16.39, 16.40_

  - [x] 15.11 复现 G 盘当前可运行 Histology baseline
    - 在 TCFM 和本地 RTX 5080 上只读重载或重训当前可运行的 Local Route A、MIL-concat、global Route A 与 classic controls，先核对固定 split/现有 protocol 预测和指标，再由新 runner 输出 baseline artifacts。
    - 单进程运行，输出仅写 `runs_v2/baselines/histology/`，保存可恢复 checkpoint、环境锁、数据/协议/model hashes；差异追加 caveat，不修改旧记录。
    - _Requirements: 1.1-1.5, 6.1-6.13, 9.9-9.26, 11.1-11.10, 13.33, 13.34_

  - [x] 15.2 实现 `hist.patient-nested5.v1` planner 和 runner wiring
    - outer/inner 都以 patient 隔离；inner-only 选择 config/epoch；冻结后每 outer fold 只评估一次。
    - runner 固定 TCFM、本地 RTX 5080 单进程、`runs_v2/strict/histology-nested/` 相对输出和 fold/epoch-boundary 可恢复 checkpoint。
    - _Requirements: 9.1-9.16, 9.39-9.41_

  - [x] 15.3 实现 `hist.lopo-fixed.v1` planner 和 runner wiring
    - 在任何 held-out labels 可访问前冻结 epoch policy，每个 patient 恰好一次完整 outer test，生成 pooled OOF metrics。
    - runner 固定 TCFM、本地 RTX 5080 单进程、`runs_v2/strict/histology-lopo/` 相对输出和 patient-boundary 可恢复 checkpoint。
    - _Requirements: 9.17-9.26, 9.39-9.41_

  - [x] 15.4 实现 Histology threshold/calibration 和 patient statistics wiring
    - primary 固定 threshold 0.5；另行标注 inner-OOF fitted threshold/temperature；连接 patient bootstrap 和 paired comparison。
    - _Requirements: 9.27-9.36_

  - [x] 15.12 以同 fold/seed/budget 小预算筛选 Histology 候选
    - 在完全相同 patient folds、seed 42、Representation_Key 对应输入和 Resource_Budget 下比较 PCA-only、RGB-only、MIL-concat、Local Route A 与 attention-disabled control；选择和 threshold/calibration 只读取 inner-patient evidence。
    - 使用 TCFM、本地 RTX 5080 单进程、`runs_v2/screening/histology/` 相对输出与可恢复 checkpoint；失败、退化和不稳定候选 append-only 保存 negative evidence。
    - _Requirements: 5.1-5.18, 9.1-9.16, 9.33-9.38, 10.10-10.25_

  - [x] 15.13 对冻结的 Histology 胜出候选执行 strict 多种子评估
    - candidate-screening checkpoint 通过后冻结候选、epoch policy、threshold、preprocessing 与预算，只对胜出候选及 matched controls 执行 seeds `42/123/7` 的 nested patient CV，并执行预冻结 epoch 的 LOPO。
    - 使用 TCFM、本地 RTX 5080 单进程、`runs_v2/strict/histology/` 相对输出与 fold/patient-boundary 可恢复 checkpoint；生成完整 OOF predictions、fold-macro/pooled metrics 与 patient uncertainty。
    - _Requirements: 9.1-9.36, 10.12-10.31, 11.1-11.31, 13.6-13.32_

  - [ ]* 15.6 编写 synthetic Histology strict integration tests
    - 使用小型 patient-group fixtures 验证 nested selection、fixed-epoch LOPO、patient OOF coverage、0.5 threshold 和 outer-label isolation。
    - _Requirements: 9.1-9.41, 16.39, 16.40_

  - [ ]* 15.7 编写属性测试：Property 17 Histology patient isolation 与 metrics
    - **Property 17: Histology patient isolation and metrics**
    - 生成 grouped patient folds，验证 group 不跨边界及所有 fold-macro/pooled OOF metrics 明确命名。
    - **Validates: Requirements 9.1-9.4, 9.9-9.26, 9.39-9.41**

  - [ ]* 15.8 编写属性测试：Property 19 threshold、calibration 和 representation labeling
    - **Property 19: Threshold, calibration, and representation labeling**
    - 验证 fitted state 只读 inner OOF、0.5 primary 保留，两个 PCA representation 和 fitted-threshold 输出不混名。
    - **Validates: Requirements 9.33-9.38**

  - [ ]* 15.9 编写 Histology statistics 与 historical-ceiling 回归测试
    - 验证 patient 10,000-resample CI、paired resampling、nested conservative metrics schema，以及历史 single-loop/fixed split/ensemble ceilings。
    - _Requirements: 1.13-1.16, 9.27-9.36, 16.24, 16.25_

- [~] 20. Checkpoint - 数据与 GPU smoke
  - 确认 TCFM、RTX 5080、G 盘两套数据、所有最小 adapter、shape/dtype、finite loss、显存、checkpoint reload 和 prediction artifact 均通过；失败时停止后续训练并询问用户。

- [~] 21. Checkpoint - baseline reproduction
  - 确认 HyperLeaf 与 Histology 的 G 盘当前可运行 baseline 已在容差内复现，所有差异与不可重载 checkpoint 已登记；失败时停止候选迭代并询问用户。

- [~] 16. Checkpoint - candidate screening
  - 确认 HyperLeaf inner-variety/LOVO seed 42 与 Histology同 fold/seed/budget 小预算筛选完成，official labels 未被用于选择，胜出候选和 negative evidence 已冻结；失败时询问用户。

- [~] 22. Checkpoint - strict multi-seed
  - 确认胜出候选及 matched controls 的 seeds 42/123/7 strict LOVO、nested patient CV 和 LOPO 完整，fold/seed/prediction 无缺失；失败时不得进入 official one-shot。

- [~] 23. Checkpoint - HyperLeaf official one-shot
  - 确认候选、preprocessing、epoch policy 和预算在 official labels 可访问前冻结，1590→820 只成功评估一次，并已保存唯一 prediction artifact 和 budget consumption record。

- [ ] 17. 在 strict 与 official one-shot 后补齐 benchmark tiers、official/proxy 和扩展候选
  - [~] 17.1 注册现有 lite models 为明确 proxy baselines
    - 将 `HybridSNLite`、`S2MambaLite`、`MamTransLite` 绑定稳定 keys、真实类名、input contracts 和含 `proxy` 的 display names。
    - _Requirements: 1.17-1.19, 6.22, 6.23, 7.5, 15.7_

  - [~] 17.2 接入 pinned Official SpectralFormer reproduction/adaptation
    - 从 `external/spectralformer/provenance.json` 建立官方来源、commit、license、source/patch hash 和 upstream reproduction record；formal adaptation 前执行 Reproduction Gate。
    - 如需训练或复现，固定 TCFM、本地 RTX 5080 单进程、工作区相对 `runs_v2/official-models/` 输出和可恢复 checkpoint；不得读取或写入 F 盘。
    - _Requirements: 6.14-6.21, 7.21-7.24, 17.25-17.33_

  - [~] 17.3 接入 S²Mamba remote-sensing sanity runner
    - 将 Indian Pines 官方协议输出为 prediction-first OA/AA/Kappa，强制 `sanity_only`，并排除 HyperLeaf/Histology rankings。
    - 如需运行，固定 TCFM、本地 RTX 5080 单进程、工作区相对 `runs_v2/sanity/` 输出和可恢复 checkpoint；不得跟随 F 盘引用。
    - _Requirements: 1.20, 7.19, 7.20, 15.8, 15.30_

  - [~] 17.4 适配获准的 official/adapted models 到任务 input contracts
    - 为 HyperLeaf/Histology 适配明确记录 resize、band selection、pooling、ignored modalities、pretraining status 和 representation keys。
    - 仅在 upstream reproduction pass 后允许 formal project comparison；任何训练固定 TCFM、本地 RTX 5080 单进程、工作区相对输出和可恢复 checkpoint。
    - _Requirements: 6.14-6.21, 7.7-7.14, 7.21-7.24_

  - [x] 17.5 实现候选空间 registry 和预先冻结配置
    - 登记 model、representation、loss、augmentation、target route、Inference_Branch、threshold 和 calibration candidates，以及 metric direction/tie-break。
    - _Requirements: 10.1-10.7_

  - [x] 17.6 实现 HyperLeaf candidate 复核配置与 controls
    - 编码 target-routed RGB/PCA consistency、independent experts、shared-head、style/strong/mixed augmentation 和 negative ablations。
    - route/consistency 选择只能使用 inner varieties；每个候选共享 dataset/protocol/seeds/folds/metrics/budget 并有 matched-input control。任何追加训练固定 TCFM、本地 RTX 5080 单进程、工作区相对输出与可恢复 checkpoint。
    - _Requirements: 10.8, 10.9, 10.12-10.25_

  - [x] 17.7 实现 Histology candidate 复核配置与 controls
    - 编码 consistency-trained PCA-only、Local Route A、MIL-concat、attention-disabled、alignment/NCE、normalization 和 negative candidates。
    - PCA-only inference branch 必须在 outer labels 可访问前由 inner patients 冻结；任何追加训练固定 TCFM、本地 RTX 5080 单进程、工作区相对输出与可恢复 checkpoint。
    - _Requirements: 10.10-10.25, 10.34_

  - [x] 17.8 实现 candidate comparison 和 promotion evaluator
    - 逐字段检查 candidate/control 的 Dataset_Hash、Protocol_Hash、Seed_Set、folds、metric config 和 Resource_Budget。
    - 缺 seed/fold/unified runner/control 时限制为 `strict_candidate`；失败只追加 negative evidence，不改变旧记录。
    - _Requirements: 10.12-10.31, 10.35-10.37_

  - [~] 17.9 实现机制描述审计器
    - 对 Global Route A、Local Route A 和模型 tier 的报告文本执行 machine-readable wording rules，阻止不受 controls 支持的归因。
    - _Requirements: 10.32-10.34, 12.19, 12.20_

  - [ ]* 17.10 编写模型适配、reproduction gate 和 candidate comparison 集成测试
    - 覆盖 proxy identity、upstream record 缺失、representation mismatch、缺 control、negative evidence 和 promotion atomicity。
    - _Requirements: 6.14-6.30, 7.6, 7.21-7.24, 10.12-10.37_

  - [ ]* 17.11 编写属性测试：Property 20 candidate promotion monotonicity
    - **Property 20: Candidate promotion monotonicity**
    - 生成候选与缺失 prerequisite，验证 selection boundary、append-only negatives 和 status 只能受 gate 提升。
    - **Validates: Requirements 10.1-10.31, 10.35-10.37**

  - [ ]* 17.12 编写属性测试：Property 21 mechanism wording safety
    - **Property 21: Mechanism wording safety**
    - 生成 Global/Local Route A report contexts，验证禁止 single-token correspondence 归因及无 control 时使用 combined-method wording。
    - **Validates: Requirements 10.32-10.34**

- [ ] 18. 完成分层测试、PowerShell 自动化编排和最终 evidence pipeline
  - [x] 18.1 建立分层 pytest suites 和 marker policy
    - 分离 `unit`、`property`、`integration`、`regression`、`smoke`、`gpu`、`full_benchmark` 和 `audit` markers；默认测试不得启动长时训练。
    - 为 TCFM 环境提供版本预检与明确 skip/fail policy，避免隐式安装或修改环境。
    - _Requirements: 16.1-16.50, 17.15-17.24_

  - [~] 18.2 实现 synthetic end-to-end integration harness
    - 串联 manifest→protocol→fit→select→refit→one-shot prediction→aggregate→claim→三格式 report，使用 toy data 和 bounded epochs。
    - _Requirements: 16.37-16.45_

  - [~] 18.3 实现 regression 和 audit fixture suite
    - 固化 outer-selector prohibition、validation-target normalization、selector mismatch、legacy ceilings、proxy/sanity ceilings 和 strict metric recomputation fixtures。
    - _Requirements: 16.20-16.27, 16.46-16.50, 17.34_

  - [x] 18.4 实现 Windows/PowerShell-safe benchmark CLI 和配置模板
    - 提供 `python -m benchmark.cli` 的 plan/run/resume/aggregate/report/audit 子命令和 PowerShell 示例；正确处理带空格路径，不使用 shell-specific chaining。
    - 暴露 TCFM environment lock、RTX 5080 device、precision、determinism、budget 和 output root 参数。
    - _Requirements: 6.31, 13.9-13.30, 14.1-14.18, 17.15-17.24_

  - [~] 18.5 实现小规模 automated smoke matrix
    - 编写 bounded synthetic/tiny-subset smoke 配置，在 CPU 与本地 RTX 5080/TCFM 上验证每个 track、model tier、resume 和 report 路径；不读取 formal outer labels。
    - GPU smoke 必须使用单进程、工作区相对 `runs_v2/smoke/` 输出和可恢复 checkpoint，先于 full benchmark 调度，并输出 machine-readable pass/fail artifact。
    - _Requirements: 5.13-5.24, 8.28-8.30, 9.39-9.41, 16.37-16.40_
  - [~] 18.6 实现完整 benchmark matrix planner 和 scheduler
    - 从冻结 dataset/protocol/model/config/Seed_Set/budget 生成 HyperLeaf official+LOVO、Histology nested+LOPO、controls、official/adapted/proxy 和 remote sanity jobs。
    - 调度器只提交 missing jobs，支持 RTX 5080 单 GPU 并发限制、checkpoint recovery 和 PowerShell 可调用的 dry-run manifest。
    - _Requirements: 7.1-7.24, 8.1-8.30, 9.1-9.41, 13.30-13.32, 14.1-14.18_

  - [~] 18.7 实现结果同步 pipeline
    - 完整 benchmark job 仅写 Run Registry 和 Prediction Artifacts；聚合后从单一 Publication Matrix 原子生成 JSON、CSV、Markdown 与 audit report。
    - 补齐 `completed_training_steps`、`input_resolution`、`pretraining_data_status`、`wall_clock_duration_ms` 和 `peak_device_memory_bytes` 资源列，并阻断 incomplete folds/seeds、hand-edited conflict、跨 protocol improvement 和跨 track overall score。
    - _Requirements: 7.7-7.14, 11.1-11.31, 12.1-12.23, 15.28-15.36_

  - [~] 18.8 实现 CVPR evidence audit command
    - 输出每个 headline/abstract candidate 的 protocol、uncertainty、limitation、Claim_Status、supporting/blocking reasons、missing provenance/control/seed/fold/prediction/reproduction fields。
    - audit 必须区分 strict、development、exploratory、sanity、proxy 和 invalid evidence，并验证所有引用 artifact hashes。
    - _Requirements: 15.1-15.38, 17.34-17.40_

  - [ ]* 18.9 编写 automated smoke matrix 测试
    - 在小样本和极低预算下验证 smoke 配置可完成、不会触发 formal evaluation budget，并可恢复中断 job。
    - _Requirements: 5.21-5.24, 13.6-13.30, 16.37-16.40_

  - [ ]* 18.10 编写 full benchmark planner/scheduler 测试
    - 只验证 dry-run job graph、dependency、missing-job difference、GPU resource limits 和不同 protocol 分区，不在测试中执行长时训练。
    - _Requirements: 3.8, 3.9, 7.15-7.18, 13.30-13.32, 14.1-14.18_

  - [ ]* 18.11 编写 result sync 与 CVPR audit 集成测试
    - 用 synthetic completed registry 验证三格式同步、四位小数、claim fields、incomplete blockers、artifact hash audit 和 task-level conclusions。
    - _Requirements: 12.1-12.23, 15.28-15.38, 17.34-17.40_

  - [ ]* 18.12 编写属性测试：Property 41 layered verification coverage
    - **Property 41: Layered verification coverage**
    - 建立 requirement-to-test manifest，验证 Requirements 16.1-16.50 每项至少映射一个 deterministic test 或至少 100 cases 的 property test。
    - **Validates: Requirements 16.1-16.50**

  - [ ]* 18.13 运行有限自动化 smoke gate 并登记结果
    - 此既有可选任务保持为后置历史验证项；若单独执行，仅通过 pytest `smoke` marker 以单次非 watch 模式登记 Environment_Lock、config hash、runtime 和 pass/fail artifact。
    - 本任务不解封或阻断 remediation、task 14.2 或 full benchmark scheduler；已完成 task 14.9/15.10 的 smoke artifacts 仅作为历史记录，不要求重新验证。
    - _Requirements: 13.1-13.8, 14.7-14.18, 16.37-16.40, 17.1-17.24_

  - [ ]* 18.14 运行冻结完整 benchmark、同步结果并生成 CVPR evidence audit
    - 使用预生成 job manifest 在 TCFM 与本地 RTX 5080 上单进程按 missing-job scheduler 执行；不得在运行中修改候选空间、folds、seeds 或预算，不得访问 F 盘。
    - 所有训练输出写工作区相对 `runs_v2/full-benchmark/` 与 `artifacts/`，保留 fold/epoch-boundary 可恢复 checkpoint；完成后从 registry 自动聚合与同步 JSON/CSV/Markdown，任一 incomplete/gate failure 保留为阻断项，不手工补数。
    - _Requirements: 7.1-7.24, 8.1-8.30, 9.1-9.41, 10.12-10.37, 11.1-11.31, 12.1-12.23, 13.1-13.36, 14.1-14.33, 15.1-15.38, 17.1-17.40_

- [ ] 24. 执行训练优先的 leakage 与 raw-lineage 修复优化波次
  - 本波次只在 `G:\Datasets\HyperCross` 工作区内运行，数据只读访问 `G:\Datasets\HyperLeaf2024` 与 `G:\Datasets\PKG - HistologyHSI-GB`；全部训练固定 TCFM、本地 RTX 5080、单进程、工作区相对 output root 和 checkpoint-boundary recovery。
  - HyperLeaf official seal 在 24.6 通过前保持关闭，任何任务均不得读取 official labels 或 `solution.csv`；旧 `src/`、`runs/`、checkpoint、float32 templates、cache 和既有 evidence 保持只读、append-only。

  - [x] 24.1 创建不可变 BF16/performance remediation 配置模板
    - 在新的 remediation 配置命名空间中复制并显式版本化所需训练参数，分别冻结 deterministic 与 performance/BF16 mode、autocast/accumulation policy、seed、预算、checkpoint interval、input-volume accounting、output root 和 environment lock；每个模板生成独立 config hash，不修改现有 float32 templates。
    - 在作业规划时验证 RTX 5080、TCFM、precision mode 与恢复 identity 完全匹配；BF16 与 float32 结果不得共享配置身份或 recovery state。
    - _Requirements: 6.4-6.13, 13.1-13.29, 14.1-14.18, 14.29-14.33, 17.1-17.24_

  - [x] 24.2 修复并执行 HyperLeaf 当前候选的 leakage-safe nested LOVO
    - 为每个 held-out variety 独立构建 selection decision；candidate、epoch 和 route 只能读取其余三个 Outer_Train varieties 的 inner folds，并在 decision 冻结后对该 held-out variety 恰好评估一次，执行 seeds `7/42/123`。
    - 禁止使用 task 14.13 已完成 outer metrics 或本次 outer metrics 调节同一 evidence claim；将 task 14.12/14.13 原结果原样保留为 `development` evidence，并在 `runs_v2/remediation/hyperleaf-nested/` 追加新 predictions、selection hashes、checkpoints、negative evidence 和 supersession links。
    - _Requirements: 1.25, 1.26, 3.3-3.9, 5.1-5.24, 8.14-8.23, 10.18-10.31, 11.1-11.23, 13.1-13.32_

  - [x] 24.3 预登记 HyperLeaf Gsw 与 target-routing 优化空间及 promotion guards
    - 在 outer evaluation 前冻结候选：`GrainWeight=PCA`；`Gsw=RGB/PCA consistency` 或 learned gated fusion；`PhiPS2=PCA/consistency`；variety-robust spectral normalization；heteroscedastic 与 target-balanced regression loss；以及每种 Representation_Key、loss 和 route 的 matched controls/ablations。
    - 将现有 Gsw `0.09112 ± 0.01013` 和相对 `0.05` control 的 `+0.000776` 仅作为设计依据，不作为新 selection score；在 hash-bound config 中预声明 meaningful/stable gate，默认要求 Gsw pooled-OOF ΔR² 至少 `+0.0100`、三个 seed 方向一致，且 GrainWeight/PhiPS2 相对 matched guard 均不得下降超过 `0.0100` R²。
    - 为所有候选固定相同 Dataset_Hash、Protocol_Hash、folds、Seed_Set、metric config、公平 Resource_Budget、precision identity、tie-break 和 recovery policy。
    - _Requirements: 4.14-4.21, 5.1-5.18, 7.6-7.14, 10.1-10.17, 10.22-10.31, 14.1-14.18_

  - [ ] 24.4 训练 HyperLeaf inner-variety-only 优化候选与 matched controls
    - 在每个 outer-variety context 中只用三个 Outer_Train varieties 的 inner folds训练、选择和冻结 candidate/epoch/route；不得查看 held-out-variety labels、task 14.13 outer metrics、official images 的 labels 或 `solution.csv`。
    - 使用 task 24.1 的独立 float32 或 BF16 hash-bound template，在 `runs_v2/optimization/hyperleaf/` 保存可恢复 checkpoint、逐样本 inner predictions、资源测量、selection decision、matched-control comparison 和 append-only negative evidence。
    - promotion 输入只允许 nested inner evidence，并执行 task 24.3 预声明的 meaningful/stable gain 与 no-target-regression guards；未过 gate 的候选保持 negative evidence，不得由 outer 或 official 分数复活。
    - _Requirements: 4.14-4.21, 5.1-5.18, 8.14-8.19, 10.1-10.31, 13.1-13.29, 14.7-14.18_

  - [ ] 24.5 对优化胜出候选执行 corrected strict HyperLeaf multi-seed LOVO
    - 仅对 task 24.4 由 nested inner evidence 选出的候选和 matched controls 执行 seeds `7/42/123`；每个 outer variety 使用独立 inner selection、完整 outer-train refit 和一次 held-out evaluation，输出到 `runs_v2/strict/hyperleaf-corrected/`。
    - 生成完整 per-fold/per-target/fold-macro/pooled-OOF predictions、seed/fold variance、资源和 recovery artifacts；corrected outer metrics 只用于最终性能估计，不得回流调整 candidate、epoch、route、loss、normalization 或 guard。
    - _Requirements: 3.3-3.9, 5.13-5.24, 8.14-8.23, 10.12-10.31, 11.1-11.23, 13.6-13.32_

  - [ ] 24.6 冻结 HyperLeaf 最终 full-train 配置并发布 corrected promotion checkpoint
    - 仅聚合 task 24.4 保存的 nested inner predictions 与预声明 tie-break/guards，冻结一个用于 1,590 样本 full-train 的 model、target routes、epoch policy、normalization、loss、precision 和 budget；不得使用 task 14.13 或 24.5 的 outer metrics选配置。
    - 校验 corrected strict folds、seeds、matched controls、provenance、prediction hashes、stable-gain/no-target-regression guards 与 recovery completeness，并原子发布 machine-readable promotion checkpoint；任一检查失败时 task 14.2 保持 blocked，official labels 与 `solution.csv` 保持 unread。
    - _Requirements: 5.1-5.24, 8.4-8.8, 10.12-10.31, 11.17-11.23, 12.11-12.16, 15.14-15.27, 17.1-17.14_

  - [ ] 24.7 生成 Histology lineage-complete fold-scoped raw-cube features
    - 以只读 streaming/chunked/memmap 访问约 `579.57 GiB` raw cubes，为每个 outer/inner fold 在许可训练 IDs 上拟合 PCA、spectral normalization 和其他 learned state；禁止把 held-out patient 数据用于 fit，并记录精确 Fit_Sample_IDs、source content hashes、chunk plan 和 transform hash。
    - 新建 hash-bound realistic input-volume budget，至少覆盖 manifest 声明的 raw bytes 与预声明 chunk overhead，不沿用旧 `8 GB` advisory budget；将 content-addressed、可复用、只读 per-fold features 写入新的 `artifacts/features/histology-raw-lineage-v2/`，不得原地修改旧 templates、spatial-mean caches 或历史 strict artifacts。
    - 在 `runs_v2/remediation/histology-raw-lineage/` 保存可恢复生成状态、完整 lineage 和资源测量；cache reuse 必须先验证 dataset/protocol/fold/source/transform hashes。
    - _Requirements: 4.1-4.13, 4.17-4.29, 9.37-9.41, 13.1-13.29, 14.1-14.28, 17.39-17.44_

  - [ ] 24.8 训练 Histology weak-patient inner-only 候选与 matched ablations
    - 在每个 outer fold 内，仅用 inner-patient evidence 预登记并比较 patient-balanced sampling、stain/RGB normalization、spectral normalization、GroupDRO/domain-invariant objectives 及对应 matched ablations，关注 P4/P6-like domain shift 但不得直接读取或调参于 held-out patient labels。
    - 每个 outer fold 保存独立 nested Selection_Decision、candidate scores/source IDs、基于 inner-patient OOF 的 patient bootstrap stability、threshold/calibration lineage 和 recovery checkpoint；primary 始终使用固定 threshold `0.5`，inner-OOF fitted threshold/temperature 仅作为明确标注的 secondary output。
    - 使用 task 24.7 的 lineage-complete features、task 24.1 的 hash-bound precision template、相同 folds/seeds/公平 budgets，并在 `runs_v2/optimization/histology-weak-patient/` 保存 matched comparisons、patient-level inner evidence 与 append-only negative results。
    - _Requirements: 4.1-4.13, 5.1-5.18, 9.1-9.8, 9.27-9.38, 10.10-10.31, 13.1-13.29_

  - [ ] 24.9 执行 Histology raw-lineage corrected strict nested CV 与 LOPO
    - 只使用 task 24.7 lineage-complete fold features 和 task 24.8 inner-only frozen decisions，对胜出候选及 matched controls 执行 seeds `7/42/123` nested patient CV，并以预冻结 epoch/config 执行 LOPO；每个 held-out patient 只在 selection freeze 后评估。
    - 在 `runs_v2/strict/histology-raw-lineage/` 生成完整 OOF predictions、固定 `0.5` primary metrics、明确标注的 secondary calibration、10,000-resample patient bootstrap、paired comparisons、资源、provenance 和 recovery artifacts。
    - 仅当 provenance、raw lineage、controls、folds、Seed_Set 和 prediction artifacts 全部通过时允许 `primary_strict`；原 cache-bound task 15.13 结果及其 incomplete lineage 原样保留为 `strict_candidate`。
    - _Requirements: 5.13-5.24, 9.1-9.41, 10.12-10.31, 11.1-11.31, 13.6-13.32, 15.14-15.27, 17.1-17.14_

- [~] 12. Checkpoint - 验证训练、prediction、metrics、claim 和 reporting 层
  - 仅在训练 fast lane、corrected remediation/optimization、strict multi-seed 和 official one-shot 完成后运行剩余 reporting/PBT；Ensure all tests pass, ask the user if questions arise.

- [~] 19. Final checkpoint - 验证自动化测试、benchmark completeness 和 CVPR evidence
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- 实现语言固定为 Python；设计中的 Pascal 风格内容仅为 structured pseudocode。
- 剩余执行顺序固定为：创建独立 hash-bound BF16/performance templates → HyperLeaf leakage-safe nested remediation 与 inner-only optimization → corrected strict LOVO 与 promotion checkpoint → Histology streaming raw-lineage features、weak-patient inner-only optimization 与 corrected strict nested/LOPO → HyperLeaf official one-shot → official/proxy、reporting/PBT、full benchmark 与论文审计。
- `G:\Datasets\HyperCross` 是唯一工作区；完全忽略 F 盘引用。数据只从 `G:\Datasets\HyperLeaf2024` 与 `G:\Datasets\PKG - HistologyHSI-GB` 读取。
- 所有训练使用 TCFM、本地 RTX 5080 单进程、工作区相对输出和可恢复 checkpoint；`src/`、`runs/`、旧 checkpoint、float32 templates 与历史 caches 只读，新候选通过 adapter/新 runner 接入。
- official test 与 `solution.csv` 保持 sealed；只有 task 24.6 corrected promotion checkpoint 通过后，task 14.2 才能在冻结配置下成功评估一次。
- 标有 `*` 的测试或执行任务是可选项，可在快速 MVP 中跳过；未标 `*` 的实现任务不可跳过。
- 每个 correctness property 都有独立 property-based test 任务，并应使用 Hypothesis 的至少 100 个生成案例，除非属性本身是固定次数统计流程。
- 默认测试命令必须是单次运行，例如 `python -m pytest -m "not full_benchmark"`；不得使用 watch mode 或启动常驻开发进程。
- 已完成的 task 14.9/15.10 smoke artifacts 仅保留为历史记录；不新增、不重跑或要求 smoke validation，task 18.5/18.13 继续作为可选后置项且不阻塞 remediation、official one-shot 或 full benchmark。规范阶段不修改代码、不运行测试或实验。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["13.7"] },
    { "id": 1, "tasks": ["13.8", "13.9"] },
    { "id": 2, "tasks": ["14.1", "15.1", "18.1", "18.4"] },
    { "id": 3, "tasks": ["14.4", "15.5"] },
    { "id": 4, "tasks": ["14.5"] },
    { "id": 5, "tasks": ["14.9", "15.10"] },
    { "id": 6, "tasks": ["14.10", "15.11"] },
    { "id": 7, "tasks": ["14.3", "15.2"] },
    { "id": 8, "tasks": ["15.3", "15.4", "17.5"] },
    { "id": 9, "tasks": ["14.11", "17.6", "17.7"] },
    { "id": 10, "tasks": ["17.8"] },
    { "id": 11, "tasks": ["14.12", "15.12"] },
    { "id": 12, "tasks": ["14.13", "15.13"] },
    { "id": 13, "tasks": ["24.1"] },
    { "id": 14, "tasks": ["24.2"] },
    { "id": 15, "tasks": ["24.3"] },
    { "id": 16, "tasks": ["24.4"] },
    { "id": 17, "tasks": ["24.5"] },
    { "id": 18, "tasks": ["24.6"] },
    { "id": 19, "tasks": ["24.7"] },
    { "id": 20, "tasks": ["24.8"] },
    { "id": 21, "tasks": ["24.9"] },
    { "id": 22, "tasks": ["14.2"] },
    { "id": 23, "tasks": ["13.2", "13.3", "17.1", "17.2", "17.3", "17.4", "17.9", "18.2", "18.3"] },
    { "id": 24, "tasks": ["18.5", "18.6", "18.7", "18.8"] },
    { "id": 25, "tasks": ["11.7", "11.10", "11.11", "11.12", "11.13", "13.4", "13.5", "13.6", "14.6", "14.7", "14.8", "15.6", "15.7", "15.8", "15.9", "17.10", "17.11", "17.12", "18.9", "18.10", "18.11", "18.12"] },
    { "id": 26, "tasks": ["18.13", "18.14"] }
  ]
}
```
