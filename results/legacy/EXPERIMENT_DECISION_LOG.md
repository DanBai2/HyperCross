# HyperCross experiment decision log

All HistologyHSI-GB values below use the same five patient folds unless stated otherwise. The primary score is fold-macro accuracy; OOF medical metrics are reported separately.

| Candidate | Result | Decision | Reason |
|---|---:|---|---|
| Global Route A on HyperLeaf | 0.5421 +/- 0.0168 R2 | Keep | Best current HyperLeaf model across three seeds |
| Local Route A on Histology | 0.9153 +/- 0.0686 accuracy | Keep | Best stable local-token model; 5-fold and strict LOPO advantage over MIL-concat |
| MIL-concat local tokens | 0.8895 +/- 0.0595 | Keep as control | Separates local-token/MIL gain from cross-modal attention |
| Official SpectralFormer CAF | 0.2586 R2 / 0.9189 accuracy | Keep as benchmark | Upstream author implementation pinned to a commit |
| MamTrans-lite | 0.8716 +/- 0.1013 accuracy over independent seeds | Keep as benchmark | High single-run value is unstable |
| Local MamTrans | 0.4913 HyperLeaf R2 / 0.9054 Histology | Negative | Dataset-specific tradeoff; no general improvement |
| Masked local pretraining | 0.4499 HyperLeaf R2 / 0.9189 Histology | Negative | No supervised gain under current pretext task |
| InstanceNorm visual encoder | 0.8961 accuracy | Negative | Removes useful morphology statistics |
| Train-patient normalization | 0.8969 accuracy | Negative | Fold-specific statistics amplify acquisition differences |
| Class-conditional alignment | 0.9057 accuracy | Negative | Over-regularizes small patient training set |
| Confidence-aware fusion | 0.8977 accuracy | Negative | Predictive entropy is unreliable for modality selection |
| Modality dropout | 0.9122 full-input accuracy | Negative robustness ablation | Does not improve full or visual-only performance |
| Domain adversarial training | 0.8994 accuracy | Negative | Too few patient domains for stable adversarial classifier |
| Patient-style consistency | 0.9054 accuracy | Negative | Artificial perturbations do not match acquisition variation |
| Patient-balanced sampling | 0.9023 accuracy | Negative | Higher variance without better held-out folds |
| GroupDRO | 0.8725 accuracy | Negative | Dynamic group weights destabilize optimization |
| Patient-episodic prototypes | 0.6647 accuracy | Negative | Support/query prototypes are too noisy and imbalanced |
| Local mask + token-NCE | 0.9122 fixed-split accuracy | Negative | Spatial mask over-constrains correspondence; below unconstrained Local Route A |
| No pooled alignment/InfoNCE | 0.9062 +/- 0.0673 five-fold accuracy | Negative | Fixed-split gain does not transfer across patients |
| Mid alignment (0.04/0.02) | 0.9056 +/- 0.0723 five-fold accuracy | Negative | Worse than default 0.08/0.04 objective |
| Leave-one-variety-out Global Route A | 0.2857 +/- 0.0707 R2 | Generalization warning | Random HyperLeaf split is substantially easier than unseen-variety transfer |
| Leave-one-variety-out domain adversarial Route A | 0.2873 +/- 0.0633 R2 | Inconclusive/negative | +0.0016 over baseline; not a reliable improvement at one adversarial weight |
| Leave-one-variety-out domain adversarial Route A (0.2) | 0.2559 +/- 0.0722 R2 | Negative | Stronger adversarial weight destabilizes regression |
| Classical SVM/RF controls | Histology 0.7904/0.7880; HyperLeaf RF 0.5271 random / 0.2845 variety-held-out R2 | Keep as controls | Confirms deep local fusion gain on Histology but small grouped HyperLeaf margin |
| Nested patient CV | Route A 0.8195 +/- 0.0925; MIL-concat 0.7897 +/- 0.0799 | Primary conservative estimate | Outer patients are never used for early stopping; pooled OOF Route A accuracy/balanced accuracy/AUROC = 0.8380/0.7942/0.8477 |
| Official S$^2$Mamba reference on Indian Pines | OA 0.9749 / AA 0.9871 / Kappa 0.9712 | Keep as remote-sensing benchmark | Official architecture/protocol, portable PyTorch selective scan; not a HyperLeaf/Histology transfer result |
| Official S$^2$Mamba adapted to HyperLeaf | 0.2854 R2 mean at epoch 17 | Keep as benchmark | 204-band cube pooled to 7x7; substantially below Global Route A 0.5421; spectral-spatial classifier is not a strong whole-leaf regression baseline |
| Mean+spatial-std spectrum Route A | 0.5477 +/- 0.0027 random-split R2; 0.2814 +/- 0.0530 variety-held-out R2 | Keep as split-specific ablation | Small random-split gain and lower seed variance, but no unseen-variety improvement |
| Corrected Global CrossModal Route A | 0.5648 +/- 0.0050 random-split R2 | Keep as corrected primary HyperLeaf result | Fixes the old --route_only model-selection bug; uses 128-bin mean spectrum and true CrossModalRouteA |
| Corrected Token Route A | 0.2767 seed-42 random-split R2 | Keep as structural ablation | Genuine patch-to-band token interaction underperforms global CrossModalRouteA on whole-leaf regression |
| Target-weighted Route A [1,2,1] | 0.2719 +/- 0.0715 variety-held-out R2 | Negative | Emphasizing Gsw does not improve unseen-variety mean R2 |
| Style-augmentation Route A | 0.3098 +/- 0.0478 variety-held-out R2 | Keep as current HyperLeaf DG configuration | RGB flip/noise and spectral noise improve mean R2 by 0.0241 and reduce variance |
| Style-augmentation Route A (random split) | 0.5226 seed-42 R2 at best epoch | DG only, not main random-split result | Below corrected no-augmentation CrossModal Route A seed-42 R2 of 0.5618 |
| HyperLeaf official train/test CrossModal Route A | 0.3663 R2 mean | Primary external benchmark; current model fails to beat classical controls | 1590 train to 820 test; fixed epoch selected on inner train-only split; PLS=0.3988 and RF=0.3735 |
| HyperLeaf official train/test style-augmented CrossModal Route A | 0.4467 +/- 0.0053 R2 mean over seeds 42/123/7 | Select as current primary HyperLeaf configuration | All three seeds beat PLS=0.3988 and RF=0.3735; per-target mean R2=0.7228/0.1198/0.4977 |
| CorrespondenceRouteA token repair | 0.2774 seed-42 random-split R2 | Negative structural ablation | Real patch/band tokens and positional pooling do not recover the global spectral-statistics advantage |
| Statistics-only fusion | 0.4699 seed-42 random-split R2 | Negative | Explicit RGB mean/std and spectral statistics alone underfit the random split |
| Statistics-residual Route A | 0.4867 seed-42 random-split R2 | Negative | Adding explicit low-order statistics to learned global features does not beat the original CrossModal Route A |
| Patient nested Route A + style augmentation | 0.8164 +/- 0.0960 accuracy; OOF AUROC 0.8504 | Negative primary configuration; retain robustness ablation | Below strict unaugmented nested mean 0.8195 +/- 0.0925, although OOF AUROC is slightly higher |
| Official HyperLeaf style augmentation + target weights [1,2,1] | 0.4401 seed-42 R2 mean | Negative; retain equal weights | Below equal-weight style-augmented official score 0.4461; Gsw weighting does not improve the external test |
| Official HyperLeaf style augmentation without alignment/NCE | 0.4478 +/- 0.0126 R2 mean over seeds 42/123/7 | Inconclusive; keep default alignment for stability | Mean is only +0.0010 over 0.4467 +/- 0.0053, while seed variance more than doubles |
| Official HyperLeaf raw absolute spectral mean | 0.0565 seed-42 R2 mean | Negative preprocessing ablation | Absolute reflectance is highly acquisition-sensitive without normalization |
| Official HyperLeaf train-only global band normalization | 0.4173 seed-42 R2 mean | Negative preprocessing ablation | Below per-image z-score + style augmentation (0.4461); train-only global statistics do not solve the shift |
| Official HyperLeaf inner-selected neural/PLSR/RF ensemble | 0.4367 seed-42 R2 mean | Negative | Blend weights selected on inner validation overfit and fall below standalone style-augmented CrossModal (0.4461) |
| Official HyperLeaf independent target heads | 0.4550 +/- 0.0075 R2 mean over seeds 42/123/7 | Select as new primary HyperLeaf configuration | Removes shared-head negative transfer; target means R2=0.7525/0.1288/0.4837 and improves over shared-head 0.4467 +/- 0.0053 |
| Official HyperLeaf shared encoder + independent heads | 0.4491 +/- 0.0079 R2 mean over seeds 42/123/7 | Keep as parameter-efficient ablation | Close to independent experts at roughly one-third encoder cost, but 0.0059 below the primary |
| Official HyperLeaf independent target heads without augmentation | 0.3316 seed-42 R2 mean | Negative | Gsw falls to -0.1380; style augmentation is required for external target-shift robustness |
| Strict leave-one-variety-out independent target experts (corrected normalization) | 0.2729 +/- 0.0711 R2 mean | No grouped improvement; retain official-test gain only | The first 0.3276 result used validation target statistics instead of outer-training statistics; corrected evaluation is close to shared encoder/heads 0.2706 |
| Strict leave-one-variety-out shared encoder + independent heads | 0.2706 +/- 0.0564 R2 mean | Negative efficiency/generalization tradeoff | Shared representation loses target-specific domain cues under unseen varieties |
| Strict leave-one-variety-out strong affine augmentation | 0.3362 +/- 0.0855 R2 mean | Grouped-only positive; do not replace official primary | RGB/spectral gain-offset perturbations improve variety holdout but official seed-42 falls to 0.4163 versus mild augmentation 0.4577 |
| Strict leave-one-variety-out mixed augmentation p=0.3 | 0.3359 +/- 0.0775 R2 mean | Grouped-only positive; retain as robustness ablation | Similar grouped gain with lower variance, but official seed-42 falls to 0.4187 |
| Strict leave-one-variety-out mixed augmentation p=0.1 | 0.3358 +/- 0.0760 R2 mean | Grouped-only positive; retain as robustness ablation | Same grouped gain, official seed-42 falls to 0.4196 |
| Official HyperLeaf unlabeled AdaBN | 0.4478 seed-42 R2 after adaptation | Negative | Updating BatchNorm with unlabeled test inputs lowers the frozen model from 0.4546 to 0.4478 |
| Official HyperLeaf unlabeled per-band feature alignment | 0.3800 seed-42 R2 after adaptation | Negative | Test-to-train mean/std mapping lowers the frozen model from 0.4547, mainly damaging GrainWeight |
| Official HyperLeaf cross-modal InfoNCE pretraining | 0.4471 seed-42 R2 mean | Negative pretraining ablation | SSL loss decreases 9.74 to 5.75, but downstream regression falls below supervised independent experts |
| Strict leave-one-variety-out independent target experts + strong affine augmentation | 0.2971 +/- 0.0697 R2 mean | Grouped-only partial improvement | Improves corrected mild independent experts (0.2729) but remains below shared strong augmentation (0.3362); target-specific heads do not solve the cross-variety shift |
| Official HyperLeaf independent target experts + strong affine augmentation | 0.4057 seed-42 R2 mean | Negative official configuration | GrainWeight/Gsw/PhiPS2 = 0.7182/0.0298/0.4693; strong perturbations damage the official test despite helping grouped validation |

## Main evidence

Strict fixed-epoch LOPO gives Local Route A 0.8806 accuracy, 0.8586 balanced accuracy and 0.8969 AUROC over 469 out-of-fold ROIs. MIL-concat gives 0.8699, 0.8462 and 0.8919. The absolute scores are lower than five-fold CV because the held-out patient is never used for early stopping.

The five-fold OOF Route A metrics are accuracy 0.9296, balanced accuracy 0.9101, AUROC 0.9027, sensitivity 0.8434, specificity 0.9769 and ECE 0.1280. The corresponding MIL-concat metrics are 0.9019, 0.8669 and 0.8824 AUROC.

The leakage-safe nested patient CV uses two training patients as an inner validation set and evaluates each outer patient fold only after early stopping is fixed. Route A obtains 0.8195 +/- 0.0925 outer-fold accuracy and pooled OOF accuracy/balanced accuracy/AUROC of 0.8380/0.7942/0.8477. MIL-concat obtains 0.7897 +/- 0.0799 and pooled 0.7996/0.7359/0.7846. These conservative values supersede the earlier single-loop five-fold numbers for claims that require strict model-selection independence.

The iterative training audit added three important controls. A local spatial attention mask with token-level correspondence NCE reached only 0.9122 on the fixed patient split, below the unconstrained Local Route A. Removing the pooled alignment and InfoNCE terms increased one fixed-split run but fell to 0.9062 +/- 0.0673 in five-fold CV; an intermediate 0.04/0.02 setting reached 0.9056 +/- 0.0723. The default 0.08/0.04 objective therefore remains selected for patient-level generalization. On HyperLeaf, leave-one-variety-out validation reduced Global Route A to 0.2857 +/- 0.0707 R2 versus 0.5421 +/- 0.0168 on the random split, so the random result must not be described as cross-variety generalization. Classical controls were also added: SVM/RF reached 0.7904/0.7880 five-fold accuracy on Histology and RF reached 0.5271 HyperLeaf R2.

The public HyperLeaf train/test protocol was then evaluated after freezing the
epoch using an inner train-only split. Training on all 1590 labeled images and
evaluating once on the 820-image test set gives CrossModal Route A 0.3663 mean
R2 (per-target 0.6684/0.0275/0.4031), below PLSR 0.3988 and RF 0.3735 under
the same protocol. This external result supersedes the random 0.5648 score as
the primary HyperLeaf claim until a domain-generalization method closes the
gap.

## Final model statement

The defensible method contribution is multimodal RGB-spectral fusion with train-only style perturbation and target-specific regression experts, evaluated under external and patient-level splits. HyperLeaf should use independent target heads as the current primary configuration; the shared-encoder/head variant is a parameter-efficient ablation. Histology claims should report both five-fold and LOPO results and include all negative ablations. The original single patient split and exploratory MamTrans ensemble must not be used as standalone SOTA evidence.
