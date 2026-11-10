# CCF A redesign and benchmark report

## Why the JACS version is insufficient for CVPR or ICML

The JACS manuscript uses two global encoders followed by bidirectional cross-attention after global pooling. In practice, each modality enters the interaction module as one sample-level vector. The paper compares mainly ResNet/DINO/XGBoost/MLP and reports application metrics, but it does not test local spatial-spectral correspondence, patient-level domain shift, modern transformer backbones, missing-modality robustness, or matched-capacity recent fusion controls.

## Route A revision

The revised model is Local Spatial-Spectral Route A. Each HistologyHSI-GB ROI is divided into a 4 x 4 grid. Every region has a visual patch token and a 128-bin local spectrum token. Bidirectional cross-attention operates on these local tokens before learned multiple-instance pooling. The model therefore represents a set of local morphology-composition relations rather than one global visual vector and one global spectral vector. The same code also supports global Route A, token Route A, gated Route A, residual Route A and multiplicative controls.

The direct numerical comparison with the JACS-style global architecture is:

The command-line selectors were audited after discovering that the historical
`--route_only` and `--token_only` flags indexed the wrong model variants. The
aliases now map explicitly to Global CrossModalRouteA and genuine TokenRouteA;
legacy rows are retained only for provenance.

| Dataset | JACS-style global Route A | Local Spatial-Spectral Route A | Relative change |
|---|---:|---:|---:|
| HyperLeaf2024 official test mean R2 | 0.4550 +/- 0.0075 (style-augmented independent target heads, 3 seeds) | 0.3988 PLSR / 0.3735 RF | official train/test result is primary; random split is diagnostic only |
| HistologyHSI-GB accuracy | 0.8243 | 0.9189 (3-seed mean) | +9.46 percentage points |

For HyperLeaf2024 the current improvement is mainly from the global cross-modal objective and stable training. The local MIL revision is strongest on HistologyHSI-GB, where local spatial heterogeneity is directly available; the negative HyperLeaf MIL result is retained as an important scope limitation.

## Modern benchmarks

The benchmark suite uses the same fixed splits, training protocol and CUDA device:

- SpectralTransformer: patchifies the continuous spectrum into 16 tokens and applies a Transformer encoder.
- SpectralFormer official CAF: upstream author implementation with group-wise band-neighbour tokens and cross-layer adaptive fusion.
- TinyViT: patchifies the RGB morphology input and applies a small vision Transformer with interpolated positional embeddings.
- FiLMFusion: conditions visual features with a learned affine transformation generated from spectral features.
- MIL-concat: matched local-token multiple-instance fusion without cross-attention.
- Global concat and the original global Route A are retained as direct controls.
- HybridSN_lite: compact 2D/1D spectral-spatial convolutional proxy.
- S2Mamba_lite: bidirectional-GRU scan proxy for spatial-spectral state-space modeling.
- MamTrans_lite: spectral Transformer plus spatial convolution with gated fusion, inspired by recent Mamba/Transformer hybrids.

Best validation results on the fixed patient-level HistologyHSI-GB split are:

| Model | Accuracy |
|---|---:|
| TinyViT | 0.7838 |
| FiLMFusion | 0.7838 |
| SpectralTransformer | 0.8378 |
| Global concat | 0.8378 |
| MIL-concat local tokens | 0.9122 |
| Local Spatial-Spectral Route A | 0.9189 |
| MamTrans_lite proxy | 0.8716 +/- 0.1013 (3 seeds; single run 0.9324) |
| Route A + MamTrans equal ensemble | 0.9392 |

On the internal HyperLeaf2024 random split, the corrected Global CrossModal Route A mean R2 is 0.5648 +/- 0.0050 over three seeds. The official 1590-train/820-test protocol gives 0.3663 without augmentation, below PLSR 0.3988 and RF 0.3735. Style augmentation with independent target heads raises the official score to 0.4550 +/- 0.0075 over seeds 42/123/7, with per-target mean R2 0.7525/0.1288/0.4837. The independent heads reduce negative transfer from the low-signal Gsw target and are now the primary HyperLeaf claim; the random score is diagnostic only. The earlier 0.5421 +/- 0.0168 row came from a pre-fix `--route_only` selector that chose the token variant; it remains in the audit trail but is no longer labeled as CrossModalRouteA. SpectralTransformer, TinyViT and FiLMFusion obtain 0.3594, 0.2438 and 0.3721, respectively; global concat obtains 0.2960.
The corrected genuine TokenRouteA reaches only 0.2767 on the same seed-42 split, so the strong HyperLeaf result is not explained by tokenization alone.

The upstream SpectralFormer CAF implementation reaches 0.2586 mean R2 on HyperLeaf2024 and 0.9189 accuracy on HistologyHSI-GB. Its official code was pinned to commit `fd08eb5edc9565f86b925a6b9abf8ebb63a5f065`; the only local addition is a small `einops` compatibility shim because that dependency is unavailable in the TCFM environment.

The additional representative HSI baselines were trained with the same fixed splits. On HyperLeaf2024, HybridSN_lite, S2Mamba_lite and MamTrans_lite reach mean R2 values of 0.0763, 0.0521 and 0.2664, respectively. On HistologyHSI-GB, the representative run gives 0.6757, 0.6757 and 0.9324 accuracy, but an independent three-seed MamTrans-lite rerun gives 0.8716 +/- 0.1013 (seeds 42, 123 and 7). The large variance makes the local Route A result (0.9189 +/- 0.0068) more reliable as the current contribution. The lite models are compact reimplementations for matched benchmarking, not claims of exact reproduction of the official HybridSN, S²Mamba or MamTrans implementations.

## What can be claimed

The strongest defensible current claim is that local spatial-spectral tokenization with Route A gives a stable 0.9189 +/- 0.0068 on the development patient split, while strict nested CV gives 0.8195 +/- 0.0925. MamTrans_lite can reach 0.9324 in one training order, but its independently reset three-seed mean is only 0.8716 +/- 0.1013; this instability must be reported and resolved with the official implementation, parameter matching and stronger validation before making a final SOTA claim. The MIL-concat control shows that part of the Route A gain comes from local tokenization and MIL pooling. The new CorrespondenceRouteA token repair reaches only 0.2774 on HyperLeaf, so cross-attention should not yet be presented as the source of the plant regression gain.

An exploratory equal-probability ensemble of the three Local Route A checkpoints and the three independently reset MamTrans-lite checkpoints reaches 0.9392 on the same 148-ROI split. This is a useful engineering result, but the ensemble weight and model family should be fixed before evaluation on a second patient split; it is not yet an independent generalization claim.

Modality-dropout training with probability 0.2 reaches 0.9122 with both modalities, 0.6757 with RGB removed and 0.9189 with spectra removed. It does not improve the full-input score or visual-only robustness in the current split, so it is retained as a negative robustness ablation rather than a claimed contribution.

An independently retrained alternate patient split (validation patients P1, P4, P6 and P9; 180 ROIs) gives 0.8667 for Local Route A, 0.6778 for MamTrans-lite and 0.8556 for their equal ensemble. This confirms that the 0.9392 ensemble result is split-sensitive and should not be presented as a general SOTA result. The paper should report both patient partitions or move to leave-one-patient-out evaluation.

The original fixed five-fold patient-level evaluation reaches 0.9153 +/- 0.0686 for Local Route A and 0.8895 +/- 0.0595 for MIL-concat, but it uses the outer fold for early stopping. It is retained as a development benchmark; the nested patient CV below is the primary conservative table.

An additional nested patient CV was run to remove outer-fold early-stopping leakage. Two patients inside each training partition were reserved for inner stopping, and the outer patients were evaluated only after checkpoint selection. Under this stricter protocol, Local Route A reaches 0.8195 +/- 0.0925 outer-fold accuracy and pooled OOF accuracy/balanced accuracy/AUROC of 0.8380/0.7942/0.8477; MIL-concat reaches 0.7897 +/- 0.0799 and 0.7996/0.7359/0.7846. These conservative numbers should replace the earlier 0.9153 figure whenever strict model-selection independence is required.

Using only out-of-fold predictions, Local Route A reaches 0.9296 accuracy, 0.9101 balanced accuracy, 0.9027 AUROC, 0.8434 sensitivity and 0.9769 specificity. MIL-concat reaches 0.9019 accuracy, 0.8669 balanced accuracy and 0.8824 AUROC. The default 0.5 threshold is already optimal for Route A in the tested threshold grid; MIL-concat improves its balanced accuracy slightly at a fixed 0.45 threshold. Route A's 10-bin ECE is 0.1280, so probability calibration remains an open issue.

Class-balanced loss was tested with the same five folds. It kept accuracy at 0.9296 but reduced balanced accuracy to 0.9047, sensitivity to 0.8193 and AUROC to 0.8957, with ECE increasing to 0.1866. It is therefore a negative ablation; the unweighted Route A objective remains the selected training protocol.

Strict leave-one-patient-out training with 20 fixed epochs gives Route A 0.8806 accuracy, 0.8586 balanced accuracy and 0.8969 AUROC over all 469 out-of-fold ROIs. The matched MIL-concat model gives 0.8699 accuracy, 0.8462 balanced accuracy and 0.8919 AUROC. The absolute scores are lower than five-fold CV because each held-out patient is never used for early stopping, but Route A preserves a measurable advantage.

An InstanceNorm visual encoder intended to reduce patient appearance shift reaches only 0.8961 +/- 0.0570 in the five-fold protocol, below the standard Route A result. This negative result suggests that simple feature normalization removes discriminative morphology and should not be used as the main domain adaptation method.

Class-conditional cross-modal alignment (lambda=0.05) reaches 0.9057 +/- 0.0593, also below standard Route A. The additional alignment objective over-regularizes the small patient-level training set; it is retained as a negative ablation.

Entropy-weighted confidence fusion (temperature 0.5) reaches 0.8977 +/- 0.0655, below standard Route A. Predictive entropy is not sufficiently reliable to decide which modality should dominate under this sample size, so this variant is also retained as a negative ablation.

Gradient-reversal patient-domain adversarial training (lambda=0.1) reaches 0.8994 +/- 0.0696, below standard Route A. With only 9--12 training patients per fold, the domain classifier is too weak and the adversarial signal removes task-relevant morphology; this is retained as a negative ablation.

Patient-style consistency augmentation (RGB affine/flip, spectral noise/band masking and symmetric KL, lambda=0.1) reaches 0.9054 +/- 0.0649, below standard Route A. The perturbations are not sufficiently realistic for the current histology acquisition process and are retained as a negative ablation.

Train-patient-only RGB and local-spectral normalization reaches 0.8969 +/- 0.0688. Although it avoids validation leakage, fold-specific normalization can amplify acquisition differences when the training patients do not represent the held-out patient; it is retained as a negative preprocessing ablation.

Equal patient-probability sampling reaches 0.9023 +/- 0.0597, below standard Route A. Reweighting small patients increases training variance without improving the held-out folds, so the default sample-level loader is retained.

GroupDRO-style dynamic patient loss reweighting (eta=0.1) reaches only 0.8725 +/- 0.0385. The rapidly changing group weights destabilize training with few patient groups, so this domain-generalization method is retained as a negative ablation.

Patient-episodic prototype regularization (prototype weight 0.2) reaches 0.6647 +/- 0.0509. The current support/query prototype construction is too noisy for the small, class-imbalanced patient groups and is retained as a negative meta-learning ablation.

## Recommended submission positioning

## Latest iterative audit

The official S$^2$Mamba and MambaHSI repositories are now pinned under
`F:\ZJU\科研\Hyperspectral\论文\HyperCross\external`. Their source trees are
available for reproduction, but the current TCFM environment lacks the
required `mamba_ssm`/selective-scan stack and the official remote-sensing data
protocol is different from the plant and biomedical cubes. Their existing
`*_lite` rows therefore remain proxies and are not reported as official SOTA.
The S$^2$Mamba architecture was nevertheless executed on the repository's
Indian Pines data using a portable reference selective scan: OA 0.9749, AA
0.9871 and Kappa 0.9712 after the official 400-epoch protocol. This is a
remote-sensing sanity benchmark; it is not a direct transfer comparison to
HyperLeaf or HistologyHSI-GB.
The same official architecture was adapted to HyperLeaf by pooling each
204-band cube to a deterministic 7x7 grid and replacing the classifier with a
three-target regression head. After 30 epochs its best mean R2 was 0.2854
(per-target [0.7698, -0.0131, 0.0994]), well below Global Route A 0.5421. This
negative result is useful evidence that a remote-sensing spectral-spatial
backbone does not automatically solve whole-leaf regression.

Restoring a second spectral statistic was also tested. Concatenating the
spatial mean and spatial standard-deviation spectra (256 values) raises the
random-split Global Route A result to 0.5477 +/- 0.0027 over three seeds, but
leave-one-variety-out falls to 0.2814 +/- 0.0530. It is therefore recorded as a
split-specific feature ablation, not as a generalization improvement.

The additional training experiments do not justify replacing the selected
configuration with a more constrained alignment model. A local attention mask
plus token-level correspondence NCE reaches 0.9122 on the fixed patient split,
below unconstrained Local Route A. Removing the pooled alignment and InfoNCE
terms improves one fixed split but drops to 0.9062 +/- 0.0673 in five-fold CV;
an intermediate alignment setting (0.04/0.02) reaches 0.9056 +/- 0.0723. The
default 0.08/0.04 objective remains the patient-level choice.

The HyperLeaf random split is also materially optimistic under a grouped
generalization check. Leave-one-variety-out Global Route A reaches only 0.2857
 +/- 0.0707 R2, compared with 0.5421 +/- 0.0168 on the random split. Classical
controls were added under the same grouped protocols: SVM/RF obtain 0.7904 /
0.7880 five-fold accuracy on Histology, while RF obtains 0.5271 random-split
HyperLeaf R2 and 0.2845 +/- 0.0262 under leave-one-variety-out validation.
The leave-one-variety-out result should be reported as a limitation and used to
motivate a future domain-generalization or pretraining contribution.
Style augmentation with independent target experts reaches corrected strict
leave-one-variety-out R2 of 0.2729 +/- 0.0711 with the epoch fixed before outer
evaluation, close to 0.2706 for a shared encoder with independent heads. The
earlier 0.3276 result used validation target statistics instead of
outer-training statistics and is removed from the claim; the 0.3098 +/- 0.0478
value also used outer-fold early stopping and remains a development comparison.
Stronger RGB contrast/brightness and spectral gain/offset perturbations raise
strict grouped R2 to 0.3362 +/- 0.0855 in the shared-head model, but reduce the
official seed-42 test to 0.4163. This remains a grouped-domain ablation; the
milder augmentation is retained for the official-test configuration.
Combining the same strong perturbations with independent target experts gives
0.2971 +/- 0.0697 under strict leave-one-variety-out and 0.4057 on the official
test (target R2 0.7182/0.0298/0.4693). The grouped gain is therefore not a
general solution: target-specific heads and aggressive acquisition shifts still
trade away official performance.
On the frozen official 1590-to-820 HyperLeaf test, the independent target
configuration reaches 0.4550 +/- 0.0075 over three seeds, above PLSR
0.3988 and RF 0.3735. It is therefore the primary HyperLeaf configuration;
the random-split 0.5226 score remains diagnostic because augmentation lowers
that split relative to 0.5648 without augmentation.
An outer-fit PCA pseudo-RGB representation raises strict leave-one-variety-out
independent experts to 0.3187 +/- 0.0212 and reduces fold variance from 0.0711
to 0.0212. An exploratory target-routed official configuration (visible RGB for
GrainWeight, asinh-transformed Gsw, PCA pseudo-RGB for PhiPS2) reaches
0.4789 +/- 0.0099 over three seeds, but the Gsw transform is negative under
strict cross-variety validation and therefore requires a nested routing study
before being claimed as the unified method.
Replacing the Gsw target transform with RGB/PCA prediction consistency gives a
more stable target-routed configuration: 0.4797 +/- 0.0040 official R2 over
three seeds. Under strict outer-fit leave-one-variety-out, the same target
routing reaches 0.3360 +/- 0.0323 across the four held-out varieties. GrainWeight
uses visible RGB, Gsw uses consistency weight 0.1, and PhiPS2 uses single-view
PCA; this is the current development candidate pending nested routing selection.
An initial variety-domain adversarial Route A reaches 0.2873 +/- 0.0633 R2,
only 0.0016 above the non-adversarial control, so it is currently
inconclusive rather than a claimed improvement. Increasing the adversarial
weight to 0.2 reduces the mean to 0.2559 +/- 0.0722, confirming that this
simple domain classifier is not a solution for the variety shift.

CVPR is the better first target because the work fits multimodal learning, medical and biological vision, explainable vision, computational imaging and datasets/evaluation. ICML would require a more general learning formulation or theory beyond the current application-specific model. The main paper should center the official HyperLeaf protocol, train-only style perturbation, target-specific regression experts, patient-level generalization, matched modern benchmarks, and reproducibility. The local token correspondence module is currently a negative ablation on HyperLeaf and should not be the headline mechanism until it improves under an external split. External SOTA numbers must only be compared when the split, preprocessing and target definition are identical.

HyperSIGMA and other remote-sensing foundation checkpoints were not inserted into the numerical table. Their published pretraining data and spectral response assumptions do not match the biomedical HistologyHSI-GB cubes or the plant HyperLeaf bands, so a direct transfer number would confound domain mismatch with architecture quality. A fair extension is to pretrain a masked local spectral-spatial encoder on the available unlabeled cubes and then fine-tune it under the same patient-level split.

## Reproducibility artifacts

All training code, split metadata, caches, histories, best checkpoints and independent evaluations are under `G:\Datasets\HyperCross`. The modern benchmark runner is `src/run_modern_benchmarks.py`; its latest output is under `runs/modern_benchmarks_20261005_232922` with separate `leaf` and `hist` subdirectories. The representative HSI runner is `src/run_hsi_representative_benchmarks.py`; the cleanest output is under `runs/hsi_representative_20261006_000656`, also with separate `leaf` and `hist` subdirectories. The masked pretraining ablation is in `src/run_masked_pretrain.py` with output `runs/masked_pretrain_20261006_025759`; the local hybrid experiments are in `src/run_local_mamtrans.py` and `src/run_globalv_local_spectral.py`. The independent MamTrans-lite stability check is in `src/run_mamtrans_hist_multiseed.py` with output `runs/mamtrans_hist_multiseed_20261006_033503`. The official SpectralFormer run is in `src/run_official_spectralformer.py` with output `runs/official_spectralformer_20261006_212147`, the modality-dropout result is in `src/run_hist_modality_dropout.py` with output `runs/hist_modality_dropout_20261006_213934`, the alternate patient split is in `src/run_hist_alt_patient_split.py` with output `runs/hist_alt_patient_20261006_214332`, the five-fold patient CV is in `src/run_hist_patient_cv.py` with output `runs/hist_patient_cv_20261006_215300`, the class-balanced ablation is in `src/run_hist_patient_cv_balanced.py` with output `runs/hist_patient_cv_balanced_20261006_220855`, strict LOPO is in `src/run_hist_lopo_fixed.py` with output `runs/hist_lopo_fixed_20261006_222020`, the InstanceNorm ablation is in `src/run_hist_patient_cv_invariant.py` with output `runs/hist_patient_cv_invariant_20261006_223235`, class-conditional alignment is in `src/run_hist_patient_cv_classalign.py` with output `runs/hist_patient_cv_classalign_20261006_224259`, confidence-aware fusion is in `src/run_hist_patient_cv_confidence.py` with output `runs/hist_patient_cv_confidence_20261006_225112`, domain adversarial training is in `src/run_hist_patient_cv_domainadv.py` with output `runs/hist_patient_cv_domainadv_20261006_225955`, patient-style consistency is in `src/run_hist_patient_cv_consistency.py` with output `runs/hist_patient_cv_consistency_20261006_230959`, train-patient normalization is in `src/run_hist_patient_cv_trainnorm.py` with output `runs/hist_patient_cv_trainnorm_20261006_232446`, patient-balanced sampling is in `src/run_hist_patient_cv_patientbalanced.py` with output `runs/hist_patient_cv_patientbalanced_20261006_233631`, GroupDRO is in `src/run_hist_patient_cv_groupdro.py` with output `runs/hist_patient_cv_groupdro_20261006_234618`, and patient-episodic prototypes are in `src/run_hist_patient_cv_protoepisodic.py` with output `runs/hist_patient_cv_protoepisodic_20261006_235618`. OOF medical metrics are in `patient_cv_metrics.json` and `balanced_cv_metrics.json`.
All training code, split metadata, caches, histories, best checkpoints and independent evaluations are under `G:\Datasets\HyperCross`. The modern benchmark runner is `src/run_modern_benchmarks.py`; its latest output is under `runs/modern_benchmarks_20261005_232922` with separate `leaf` and `hist` subdirectories. The representative HSI runner is `src/run_hsi_representative_benchmarks.py`; the cleanest output is under `runs/hsi_representative_20261006_000656`, also with separate `leaf` and `hist` subdirectories. The masked pretraining ablation is in `src/run_masked_pretrain.py` with output `runs/masked_pretrain_20261006_025759`; the local hybrid experiments are in `src/run_local_mamtrans.py` and `src/run_globalv_local_spectral.py`. The independent MamTrans-lite stability check is in `src/run_mamtrans_hist_multiseed.py` with output `runs/mamtrans_hist_multiseed_20261006_033503`. The official SpectralFormer run is in `src/run_official_spectralformer.py` with output `runs/official_spectralformer_20261006_212147`, the modality-dropout result is in `src/run_hist_modality_dropout.py` with output `runs/hist_modality_dropout_20261006_213934`, the alternate patient split is in `src/run_hist_alt_patient_split.py` with output `runs/hist_alt_patient_20261006_214332`, the five-fold patient CV is in `src/run_hist_patient_cv.py` with output `runs/hist_patient_cv_20261006_215300`, the class-balanced ablation is in `src/run_hist_patient_cv_balanced.py` with output `runs/hist_patient_cv_balanced_20261006_220855`, strict LOPO is in `src/run_hist_lopo_fixed.py` with output `runs/hist_lopo_fixed_20261006_222020`, the InstanceNorm ablation is in `src/run_hist_patient_cv_invariant.py` with output `runs/hist_patient_cv_invariant_20261006_223235`, class-conditional alignment is in `src/run_hist_patient_cv_classalign.py` with output `runs/hist_patient_cv_classalign_20261006_224259`, confidence-aware fusion is in `src/run_hist_patient_cv_confidence.py` with output `runs/hist_patient_cv_confidence_20261006_225112`, domain adversarial training is in `src/run_hist_patient_cv_domainadv.py` with output `runs/hist_patient_cv_domainadv_20261006_225955`, patient-style consistency is in `src/run_hist_patient_cv_consistency.py` with output `runs/hist_patient_cv_consistency_20261006_230959`, train-patient normalization is in `src/run_hist_patient_cv_trainnorm.py` with output `runs/hist_patient_cv_trainnorm_20261006_232446`, patient-balanced sampling is in `src/run_hist_patient_cv_patientbalanced.py` with output `runs/hist_patient_cv_patientbalanced_20261006_233631`, GroupDRO is in `src/run_hist_patient_cv_groupdro.py` with output `runs/hist_patient_cv_groupdro_20261006_234618`, patient-episodic prototypes are in `src/run_hist_patient_cv_protoepisodic.py` with output `runs/hist_patient_cv_protoepisodic_20261006_235618`, and the consolidated decision log is `EXPERIMENT_DECISION_LOG.md`. OOF medical metrics are in `patient_cv_metrics.json` and `balanced_cv_metrics.json`.
