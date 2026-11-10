# Current HSI model families and deployment status

This note records the recent model families considered for a CVPR/ICML-oriented revision. The literature list is used to select representative controls; it is not a claim that every paper is directly comparable to the two biomedical/plant datasets here.

| Family | Core idea | Source | Status in this workspace |
|---|---|---|---|
| SpectralFormer | Local group-wise spectral embeddings, Transformer sequence modeling and cross-layer skips | [arXiv:2107.02988](https://arxiv.org/abs/2107.02988) | Upstream CAF code pinned and trained on both datasets; mean R2 0.2586 / accuracy 0.9189 |
| HybridSN | 3-D spectral-spatial feature extraction followed by 2-D convolution | [IEEE TGRS, 2020](https://doi.org/10.1109/TGRS.2020.2993204) | `HybridSN_lite` compact proxy, trained on both datasets |
| S²Mamba / spectral-spatial Mamba | State-space scans over spatial and spectral axes with gated fusion | [S²Mamba arXiv:2404.18213](https://arxiv.org/abs/2404.18213), [Spectral-Spatial Mamba arXiv:2404.18401](https://arxiv.org/abs/2404.18401) | `S2Mamba_lite` bidirectional-GRU scan proxy, trained on both datasets |
| MamTrans | Mamba/Transformer hybrid with bidirectional sequence modeling | [MamTrans](https://www.sciencedirect.com/science/article/pii/S0165168424002895) | `MamTrans_lite` spectral Transformer + spatial convolution proxy, trained on both datasets |
| Masked HSI ViT | Masked spectral-spatial pretraining followed by fine-tuning | [CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Scheibenreif_Masked_Vision_Transformers_for_Hyperspectral_Image_Classification_CVPRW_2023_paper.pdf) | Local masked reconstruction implemented; no gain in the first ablation |
| HyperSIGMA | Remote-sensing hyperspectral foundation model with sparse sampling attention and large-scale pretraining | [arXiv:2406.11519](https://arxiv.org/abs/2406.11519) | Not used for a numerical claim because its pretraining domain and wavelength assumptions do not match these datasets |
| Official S$^2$Mamba | Official spatial-spectral state-space model with spatial and spectral scans | [GitHub](https://github.com/PURE-melo/S2Mamba) | Reproduced on bundled Indian Pines with portable reference scan: OA 0.9749 / AA 0.9871 / Kappa 0.9712; not transferred to plant/biomedical data |
| MambaHSI | Official image-level spatial-spectral Mamba backbone | [GitHub](https://github.com/li-yapeng/MambaHSI) | Source pinned; official environment requires PyTorch 1.13/CUDA 11.7/mamba-ssm and remains pending |

## Benchmark outcomes

The fixed HyperLeaf random 80/20 split gives mean R2 of 0.0763 (`HybridSN_lite`), 0.0521 (`S2Mamba_lite`) and 0.2664 (`MamTrans_lite`). After fixing the CLI selector so that `--crossmodal_only` truly instantiates Global CrossModal Route A, the corrected three-seed result is 0.5648 +/- 0.0050. On the official 1,590-to-820 train/test protocol, style augmentation with independent target heads reaches 0.4550 +/- 0.0075 over three seeds, above PLSR 0.3988 and RF 0.3735. The earlier 0.5421 +/- 0.0168 value is retained as a legacy token-variant audit row.

The official S$^2$Mamba architecture was adapted to the same HyperLeaf split by pooling each 204-band cube to 7x7 and replacing the classifier with a three-target regression head. Its best 30-epoch mean R2 is 0.2854 (per-target [0.7698, -0.0131, 0.0994]), below Global Route A. This confirms that the current advantage is specific to whole-leaf multimodal regression rather than a generic Mamba backbone effect.

The fixed HistologyHSI-GB patient split gives accuracy of 0.6757, 0.6757 and 0.9324 in the first representative run. After resetting the random seed independently for each run, MamTrans_lite gives 0.9257, 0.7297 and 0.9595 (mean 0.8716 +/- 0.1013). Local MIL Route A gives 0.9189 +/- 0.0068 over three seeds. The discrepancy is a useful warning: the revised paper must report seed sensitivity, verify the official architecture and parameter budget, and avoid an unconditional SOTA statement until multi-seed and external-split tests are complete.

The first masked local spectral-spatial pretraining ablation did not improve the supervised task: HyperLeaf mean R2 was 0.4524 from scratch versus 0.4499 after transfer, and Histology accuracy was 0.9189 for both. A local MamTrans-style model improved HyperLeaf to 0.4913 but reached only 0.9054 on Histology. A global visual plus local spectral Transformer reached 0.4808 and 0.8986, respectively. These results support a dataset-specific design: global Route A for HyperLeaf and stable local MIL Route A for Histology, while masked pretraining and naive local Transformer fusion should remain negative ablations until their objectives are redesigned.

Modality dropout at probability 0.2 reached 0.9122 with both modalities, 0.6757 with RGB removed and 0.9189 with spectra removed. It did not improve full-input performance, but the missing-modality evaluation is retained because CVPR multimodal reviewers commonly require this robustness analysis.

On an independently retrained alternate patient split (P1, P4, P6 and P9 held out; 180 ROIs), Local Route A reached 0.8667, MamTrans-lite 0.6778 and their equal ensemble 0.8556. This split sensitivity is evidence that the original four-patient split is not sufficient for a final claim.

The five-fold patient-level evaluation over all 469 ROIs gives 0.9153 +/- 0.0686 for Local Route A and 0.8895 +/- 0.0595 for matched MIL-concat. The 2.58-point average gain is the strongest current evidence that local cross-modal interaction adds value beyond local tokenization, although fold variance remains substantial.

That single-loop five-fold result uses the outer fold for early stopping and is therefore a development result. The leakage-safe nested patient CV uses two inner patients for checkpoint selection and leaves the outer patients untouched until final evaluation: Route A is 0.8195 +/- 0.0925 outer-fold accuracy and MIL-concat is 0.7897 +/- 0.0799; pooled OOF Route A accuracy/balanced accuracy/AUROC are 0.8380/0.7942/0.8477. These nested values are the conservative numbers for a final manuscript.

Out-of-fold medical metrics are: Route A accuracy 0.9296, balanced accuracy 0.9101, AUROC 0.9027, sensitivity 0.8434, specificity 0.9769 and ECE 0.1280; MIL-concat accuracy 0.9019, balanced accuracy 0.8669 and AUROC 0.8824. This supports reporting AUROC and balanced accuracy alongside accuracy and motivates post-hoc calibration.

Class-balanced loss was tested across the same five folds. Accuracy stayed at 0.9296, but balanced accuracy fell to 0.9047 and sensitivity to 0.8193, with ECE worsening to 0.1866. The unweighted objective is retained.

Strict fixed-epoch leave-one-patient-out evaluation gives Route A 0.8806 accuracy, 0.8586 balanced accuracy and 0.8969 AUROC, versus MIL-concat 0.8699, 0.8462 and 0.8919. The lower absolute value is expected because no held-out patient is used for early stopping; the retained advantage is therefore more conservative evidence.

Adding InstanceNorm to the visual encoder reduced five-fold accuracy to 0.8961 +/- 0.0570, so simple appearance normalization is retained as a negative domain-shift ablation.

Class-conditional cross-modal alignment with lambda=0.05 reached 0.9057 +/- 0.0593, below the unmodified Route A. It is retained as a negative regularization ablation.

Entropy-weighted confidence fusion at temperature 0.5 reached 0.8977 +/- 0.0655, below standard Route A. The modality confidence estimates are not reliable enough on this small dataset.

Gradient-reversal patient-domain adversarial training with lambda=0.1 reached 0.8994 +/- 0.0696, below standard Route A. The available patient count is too small for a stable domain classifier in this protocol.

Patient-style consistency augmentation reached 0.9054 +/- 0.0649, below standard Route A. The current perturbations are not sufficiently acquisition-specific and are retained as a negative augmentation ablation.

Train-patient-only RGB/spectral normalization reached 0.8969 +/- 0.0688, so fold-specific preprocessing is also retained as a negative ablation.

Equal patient-probability sampling reached 0.9023 +/- 0.0597, below standard Route A; the default sample-level loader is retained.

GroupDRO-style dynamic patient reweighting reached 0.8725 +/- 0.0385 and was unstable across folds; it is retained as a negative domain-generalization ablation.

Patient-episodic prototype regularization reached 0.6647 +/- 0.0509 and was highly unstable, so the current prototype construction is retained as a negative meta-learning ablation.

## Recommended next experiment

If pretraining is revisited, mask contiguous wavelength groups and spatial tiles rather than independently masking already-compressed tokens, and add a cross-modal consistency or contrastive objective. Fine-tune with patient-level HistologyHSI-GB splits and sample-level HyperLeaf regression, reporting linear-probe, full fine-tune, missing-modality and cross-patient tests.
