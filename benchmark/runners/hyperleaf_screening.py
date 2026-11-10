"""Seed-42, train-only grouped screening for staged task 14.12.

The executor reads the frozen HyperLeaf training manifest, ``train.csv``, and
only the training cubes named by that manifest.  It has no official-test or
solution-label code path.  Candidate and epoch selection use exact four-fold
leave-one-variety-out out-of-fold predictions from the training partition.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS, HYPERLEAF_VARIETIES
from benchmark.catalog.raw_cube import open_tiff_cube
from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import PredictionArtifactWriter
from benchmark.models.hyperleaf_candidates import (
    AVERAGE_BRANCH,
    PCA_BRANCH,
    RGB_BRANCH,
    TARGET_ROUTED_BRANCH,
    RGB_PCA_CONSISTENCY_MODEL_KEY,
    create_hyperleaf_candidate_registry,
    rgb_pca_consistency_loss,
)
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.runners.hyperleaf_candidates import (
    GSW_CANDIDATE_CONFIGS,
    HYPERLEAF_CANDIDATE_SPECS,
    HyperLeafCandidateBranch,
)
from benchmark.schema import (
    FitBoundary,
    FitBoundaryScope,
    MetricDirection,
    PredictionKey,
    PredictionRecord,
    ResourceField,
    RuntimeMeasurements,
    SampleManifest,
    TaskKind,
    MetricSchema,
    assess_resource_budget,
    canonical_serialize,
    hash_file,
    hash_object,
)
from benchmark.transforms.hyperleaf_candidates import (
    GswLossKind,
    GswTargetTransform,
    HYPERLEAF_STYLE_CONFIGS,
    apply_hyperleaf_style,
    gsw_regression_loss,
    inverse_gsw_target,
    transform_gsw_target,
)
from benchmark.transforms.preprocessing import PreprocessingManager


WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
TRAIN_MANIFEST = WORKSPACE / "artifacts" / "manifests" / "hyperleaf2024" / "66b78e98dca72c272f5dd764e2c72b2543e4be4c7f47a49bf97a01fa8714efd1.json"
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "screening.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "screening" / "hyperleaf"
PROTOCOL_KEY = "leaf.train-inner-lovo.screening.v1"
SEED = 42
BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
RUNG_EPOCHS = (20, 40, 80)
RUNG_SURVIVORS = (8, 4, 1)
TARGET_ROUTES = (RGB_BRANCH, AVERAGE_BRANCH, PCA_BRANCH)


@dataclass(frozen=True)
class ScreeningCandidate:
    candidate_id: str
    family: str
    model_key: str
    inference_branch: str
    style_key: str
    loss_kind: str
    target_transform: str
    consistency_weight: float
    target_routes: tuple[str, ...] = ()

    @property
    def safe_id(self) -> str:
        return hashlib.sha256(self.candidate_id.encode("utf-8")).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return hash_object(asdict(self))


@dataclass(frozen=True)
class ScreeningFold:
    fold_id: str
    validation_variety: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]


@dataclass
class FoldTrainingState:
    fold: ScreeningFold
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    completed_steps: int = 0


@dataclass(frozen=True)
class ScreeningData:
    sample_ids: tuple[str, ...]
    id_to_index: Mapping[str, int]
    varieties: tuple[str, ...]
    targets: np.ndarray
    rgb: np.ndarray
    spectrum: np.ndarray
    pca_by_fold: Mapping[str, np.ndarray]
    target_artifacts: Mapping[str, Any]
    pca_artifacts: Mapping[str, Any]
    input_volume_bytes: int
    train_csv_hash: str


def build_screening_candidates() -> tuple[ScreeningCandidate, ...]:
    """Return the complete task-14.11 branch and Gsw candidate declarations."""

    candidates: list[ScreeningCandidate] = []
    for branch in HyperLeafCandidateBranch:
        spec = HYPERLEAF_CANDIDATE_SPECS[branch]
        candidates.append(
            ScreeningCandidate(
                candidate_id=f"branch::{branch.value}",
                family="grouped-branch-control",
                model_key=spec.model_key,
                inference_branch=spec.inference_branch,
                style_key=spec.style_key,
                loss_kind=GswLossKind.MSE.value,
                target_transform=GswTargetTransform.STANDARDIZED.value,
                consistency_weight=spec.consistency_weight,
                target_routes=TARGET_ROUTES if spec.inference_branch == TARGET_ROUTED_BRANCH else (),
            )
        )
    for config in GSW_CANDIDATE_CONFIGS:
        candidates.append(
            ScreeningCandidate(
                candidate_id=config.candidate_id,
                family="gsw-specialist",
                model_key=config.model_key,
                inference_branch=config.inference_branch,
                style_key=config.style_key,
                loss_kind=config.loss_kind.value,
                target_transform=config.target_transform.value,
                consistency_weight=config.consistency_weight,
            )
        )
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id.encode("utf-8")))
    if len(ordered) != 4 + len(GSW_CANDIDATE_CONFIGS):
        raise RuntimeError("screening candidate set is incomplete")
    if len({item.candidate_id for item in ordered}) != len(ordered):
        raise RuntimeError("screening candidate IDs are not unique")
    if any(item.style_key not in HYPERLEAF_STYLE_CONFIGS for item in ordered):
        raise RuntimeError("screening candidate uses an undeclared style")
    return ordered


def build_inner_lovo_folds(
    sample_ids: Sequence[str], varieties: Sequence[str]
) -> tuple[ScreeningFold, ...]:
    """Create exact train-partition inner LOVO folds, one validation variety each."""

    if len(sample_ids) != len(varieties) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs and varieties must be aligned and unique")
    by_variety = {name: [] for name in HYPERLEAF_VARIETIES}
    for sample_id, variety in zip(sample_ids, varieties):
        if variety not in by_variety:
            raise ValueError(f"unexpected HyperLeaf variety: {variety}")
        by_variety[variety].append(sample_id)
    if any(not values for values in by_variety.values()):
        raise ValueError("every HyperLeaf variety must be represented")
    all_ids = set(sample_ids)
    folds = []
    for variety in HYPERLEAF_VARIETIES:
        validation = tuple(sorted(by_variety[variety], key=str.encode))
        train = tuple(sorted(all_ids - set(validation), key=str.encode))
        folds.append(
            ScreeningFold(
                fold_id=f"inner-lovo-{variety.casefold()}",
                validation_variety=variety,
                train_ids=train,
                validation_ids=validation,
            )
        )
    observed = [sample_id for fold in folds for sample_id in fold.validation_ids]
    if len(observed) != len(set(observed)) or set(observed) != all_ids:
        raise RuntimeError("inner LOVO folds do not provide exact OOF coverage")
    return tuple(folds)


def compute_grouped_metrics(
    sample_ids: Sequence[str],
    fold_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute per-fold, pooled, and fold-macro target metrics from exact OOF rows."""

    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    expected = (len(sample_ids), len(HYPERLEAF_TARGETS))
    if truth.shape != expected or prediction.shape != expected:
        raise ValueError(f"OOF arrays must have shape {expected}")
    if len(fold_ids) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("OOF identity arrays must be aligned and unique")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("OOF predictions must be finite")

    def target_metrics(indices: np.ndarray) -> dict[str, dict[str, float | None]]:
        result: dict[str, dict[str, float | None]] = {}
        for index, target in enumerate(HYPERLEAF_TARGETS):
            actual = truth[indices, index]
            forecast = prediction[indices, index]
            total = float(np.sum((actual - actual.mean()) ** 2, dtype=np.float64))
            residual = float(np.sum((actual - forecast) ** 2, dtype=np.float64))
            result[target] = {
                "r2": None if total == 0.0 else 1.0 - residual / total,
                "mae": float(np.mean(np.abs(actual - forecast), dtype=np.float64)),
            }
        return result

    pooled = target_metrics(np.arange(len(sample_ids)))
    per_fold: dict[str, Any] = {}
    for fold_id in sorted(set(fold_ids), key=str.encode):
        indices = np.asarray([i for i, value in enumerate(fold_ids) if value == fold_id])
        per_fold[fold_id] = target_metrics(indices)
    fold_macro: dict[str, Any] = {}
    for target in HYPERLEAF_TARGETS:
        r2_values = [per_fold[fold][target]["r2"] for fold in per_fold]
        mae_values = [float(per_fold[fold][target]["mae"]) for fold in per_fold]
        fold_macro[target] = {
            "r2": None if any(value is None for value in r2_values) else float(np.mean(r2_values)),
            "mae": float(np.mean(mae_values)),
        }
    valid_r2 = [pooled[target]["r2"] for target in HYPERLEAF_TARGETS]
    return {
        "primary_objective": "pooled_oof.r2.Gsw",
        "pooled_oof": {
            "targets": pooled,
            "mean_r2": None if any(value is None for value in valid_r2) else float(np.mean(valid_r2)),
            "mean_mae": float(np.mean([pooled[target]["mae"] for target in HYPERLEAF_TARGETS])),
        },
        "per_fold": per_fold,
        "fold_macro": {
            "targets": fold_macro,
            "mean_r2": None
            if any(fold_macro[target]["r2"] is None for target in HYPERLEAF_TARGETS)
            else float(np.mean([fold_macro[target]["r2"] for target in HYPERLEAF_TARGETS])),
            "mean_mae": float(np.mean([fold_macro[target]["mae"] for target in HYPERLEAF_TARGETS])),
        },
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _atomic_torch(path: Path, value: Any) -> None:
    stream = io.BytesIO()
    torch.save(value, stream)
    _atomic_bytes(path, stream.getvalue())


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _json_bytes(value)
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _tree_fingerprint(root: Path) -> str:
    entries = []
    if root.exists():
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")):
            stat = path.stat()
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "file": path.is_file(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return hash_object(entries)


def _set_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _load_train_manifest() -> SampleManifest:
    payload = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    manifest = SampleManifest.model_validate(payload)
    if manifest.dataset_hash != TRAIN_MANIFEST.stem or len(manifest.records) != 1590:
        raise RuntimeError("frozen HyperLeaf training manifest identity is invalid")
    for record in manifest.records:
        metadata = record.label_schema_metadata
        if metadata.get("partition") != "train" or metadata.get("labels_available") is not True:
            raise RuntimeError("screening manifest contains a non-training record")
        if record.group_id not in HYPERLEAF_VARIETIES:
            raise RuntimeError("screening manifest contains an invalid training variety")
        if not str(record.label_reference).startswith("labels://hyperleaf2024/train/"):
            raise RuntimeError("screening manifest contains a non-training label reference")
    return manifest


def _load_training_table(manifest: SampleManifest) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    frame = pd.read_csv(DATA_ROOT / "train.csv")
    if len(frame) != 1590 or any(target not in frame for target in HYPERLEAF_TARGETS):
        raise RuntimeError("HyperLeaf training table contract differs from the frozen manifest")
    sample_ids = tuple(str(int(value)).zfill(5) for value in frame["ImageId"])
    if set(sample_ids) != set(manifest.canonical_order):
        raise RuntimeError("train.csv IDs differ from the frozen training manifest")
    rows = frame.assign(_sample_id=sample_ids).set_index("_sample_id")
    ordered_ids = manifest.canonical_order
    targets = rows.loc[list(ordered_ids), list(HYPERLEAF_TARGETS)].to_numpy(dtype=np.float64)
    record_groups = {record.sample_id: str(record.group_id) for record in manifest.records}
    varieties = tuple(record_groups[sample_id] for sample_id in ordered_ids)
    if not np.isfinite(targets).all():
        raise RuntimeError("training targets contain non-finite values")
    return ordered_ids, varieties, targets


def _fit_fold_preprocessing(
    folds: Sequence[ScreeningFold],
    sample_ids: Sequence[str],
    targets: np.ndarray,
    raw_spectra: np.ndarray,
    dataset_hash: str,
    protocol_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    manager = PreprocessingManager()
    target_artifacts: dict[str, Any] = {}
    pca_artifacts: dict[str, Any] = {}
    for fold in folds:
        train_indices = np.asarray([index[sample_id] for sample_id in fold.train_ids])
        boundary = FitBoundary(
            fold_id=fold.fold_id,
            scope=FitBoundaryScope.INNER_TRAIN,
            inner_fold_id=fold.fold_id,
            allowed_fit_sample_ids=fold.train_ids,
            outer_test_sample_ids=fold.validation_ids,
        )
        target_artifacts[fold.fold_id] = manager.fit_target_normalization(
            targets[train_indices],
            fold.train_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
        )
        pca_artifacts[fold.fold_id] = manager.fit_pca(
            raw_spectra[train_indices],
            fold.train_ids,
            boundary,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
            n_components=3,
            whiten=False,
        )
    return target_artifacts, pca_artifacts


def _resize_chw(values: np.ndarray, size: tuple[int, int] = (32, 64)) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy().astype(np.float32, copy=False)


def _load_representations(
    manifest: SampleManifest,
    sample_ids: Sequence[str],
    folds: Sequence[ScreeningFold],
    targets: np.ndarray,
    protocol_hash: str,
    execution_root: Path,
) -> ScreeningData:
    records = {record.sample_id: record for record in manifest.records}
    raw_spectra = []
    input_volume = (DATA_ROOT / "train.csv").stat().st_size
    for sample_id in sample_ids:
        reference = records[sample_id].input_references["full_cube"]
        path = DATA_ROOT / Path(reference)
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError("training cube left the approved G: data root")
        input_volume += path.stat().st_size
        with open_tiff_cube(path, axis_order="BHW") as reader:
            raw = np.asarray(reader.memmap_view(), dtype=np.float64).mean(axis=(1, 2))
        if not np.isfinite(raw).all():
            raise RuntimeError(f"training cube contains non-finite values: {sample_id}")
        raw_spectra.append(raw)
    raw_spectrum_array = np.stack(raw_spectra)
    target_artifacts, pca_artifacts = _fit_fold_preprocessing(
        folds,
        sample_ids,
        targets,
        raw_spectrum_array,
        manifest.dataset_hash,
        protocol_hash,
    )

    transforms_root = execution_root / "transforms"
    for fold in folds:
        _atomic_bytes(transforms_root / fold.fold_id / "target-normalization.canonical.json", canonical_serialize(target_artifacts[fold.fold_id]))
        _atomic_bytes(transforms_root / fold.fold_id / "pca.canonical.json", canonical_serialize(pca_artifacts[fold.fold_id]))

    rgb_rows: list[np.ndarray] = []
    spectrum_rows: list[np.ndarray] = []
    pca_rows: dict[str, list[np.ndarray]] = {fold.fold_id: [] for fold in folds}
    manager = PreprocessingManager()
    for position, sample_id in enumerate(sample_ids):
        path = DATA_ROOT / Path(records[sample_id].input_references["full_cube"])
        with open_tiff_cube(path, axis_order="BHW") as reader:
            cube_native = reader.memmap_view()
            bands = reader.shape[0]
            band_indices = tuple(min(bands - 1, int(bands * ratio)) for ratio in (0.72, 0.48, 0.20))
            rgb = np.stack([np.asarray(cube_native[index], dtype=np.float32) for index in band_indices])
            rgb = rgb / (65535.0 if float(np.nanmax(rgb)) > 1.0 else 1.0)
            image = Image.fromarray(np.moveaxis((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 0, -1))
            resized = image.resize((64, 32), Image.Resampling.BILINEAR)
            rgb_rows.append(np.moveaxis(np.asarray(resized, dtype=np.float32) / 255.0, -1, 0))

            raw = raw_spectrum_array[position]
            interpolated = np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, raw.size), raw)
            scale = float(interpolated.std())
            spectrum_rows.append(((interpolated - interpolated.mean()) / (scale if scale > 1e-12 else 1.0)).astype(np.float32))

            pixels = np.moveaxis(np.asarray(cube_native, dtype=np.float32), 0, -1).reshape(-1, bands)
            for fold in folds:
                boundary = FitBoundary(
                    fold_id=fold.fold_id,
                    scope=FitBoundaryScope.INNER_TRAIN,
                    inner_fold_id=fold.fold_id,
                    allowed_fit_sample_ids=fold.train_ids,
                    outer_test_sample_ids=fold.validation_ids,
                )
                projected = manager.apply(pca_artifacts[fold.fold_id], pixels, boundary)
                projected = projected.reshape(reader.shape[1], reader.shape[2], 3)
                minimum = projected.min((0, 1), keepdims=True)
                maximum = projected.max((0, 1), keepdims=True)
                denominator = np.where(maximum - minimum > 1e-12, maximum - minimum, 1.0)
                chw = np.moveaxis(((projected - minimum) / denominator).astype(np.float32), -1, 0)
                pca_rows[fold.fold_id].append(_resize_chw(chw))
        if (position + 1) % 100 == 0 or position + 1 == len(sample_ids):
            print(json.dumps({"stage": "representation_materialization", "completed": position + 1, "total": len(sample_ids)}), flush=True)

    rgb_array = np.stack(rgb_rows).astype(np.float32)
    spectrum_array = np.stack(spectrum_rows).astype(np.float32)
    pca_arrays = {fold_id: np.stack(rows).astype(np.float32) for fold_id, rows in pca_rows.items()}
    if rgb_array.shape != (1590, 3, 32, 64) or spectrum_array.shape != (1590, 128):
        raise RuntimeError("materialized train representation shape is invalid")
    if any(values.shape != (1590, 3, 32, 64) for values in pca_arrays.values()):
        raise RuntimeError("materialized fold PCA representation shape is invalid")
    if not np.isfinite(rgb_array).all() or not np.isfinite(spectrum_array).all() or any(not np.isfinite(value).all() for value in pca_arrays.values()):
        raise RuntimeError("materialized train representation contains non-finite values")
    return ScreeningData(
        sample_ids=tuple(sample_ids),
        id_to_index={sample_id: index for index, sample_id in enumerate(sample_ids)},
        varieties=tuple(records[sample_id].group_id or "" for sample_id in sample_ids),
        targets=targets,
        rgb=rgb_array,
        spectrum=spectrum_array,
        pca_by_fold=pca_arrays,
        target_artifacts=target_artifacts,
        pca_artifacts=pca_artifacts,
        input_volume_bytes=input_volume,
        train_csv_hash=hash_file(DATA_ROOT / "train.csv"),
    )


def _replace_adaptive_pool(module: nn.Module) -> int:
    """Replace the candidate's deterministic 16-to-8 average pooling equivalent."""

    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool1d) and child.output_size == 8:
            setattr(module, name, nn.AvgPool1d(kernel_size=2, stride=2))
            replaced += 1
        else:
            replaced += _replace_adaptive_pool(child)
    return replaced


def _build_model(candidate: ScreeningCandidate, device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = create_hyperleaf_candidate_registry().build(candidate.model_key)
    _replace_adaptive_pool(model)
    return model.to(device=device, dtype=torch.float32)


def _target_space(values: Tensor, candidate: ScreeningCandidate) -> Tensor:
    result = values.clone()
    result[:, 1] = transform_gsw_target(result[:, 1], GswTargetTransform(candidate.target_transform))
    return result


def _task_loss(prediction: Tensor, targets: Tensor, candidate: ScreeningCandidate) -> Tensor:
    other = (F.mse_loss(prediction[:, 0], targets[:, 0]) + F.mse_loss(prediction[:, 2], targets[:, 2])) / 2.0
    gsw = gsw_regression_loss(
        prediction[:, 1],
        targets[:, 1],
        loss_kind=GswLossKind(candidate.loss_kind),
    )
    return (2.0 * other + gsw) / 3.0


def _forward(
    state: FoldTrainingState,
    candidate: ScreeningCandidate,
    rgb: Tensor,
    pca: Tensor,
    spectrum: Tensor,
    *,
    targets: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if candidate.model_key == RGB_PCA_CONSISTENCY_MODEL_KEY:
        prediction, branches = state.model(
            rgb,
            pca,
            spectrum,
            inference_branch=candidate.inference_branch,
            target_routes=candidate.target_routes or None,
            return_branches=True,
        )
        if targets is not None:
            branch_loss = (
                _task_loss(branches[RGB_BRANCH], targets, candidate)
                + _task_loss(branches[PCA_BRANCH], targets, candidate)
            ) / 2.0
            penalty = rgb_pca_consistency_loss(
                branches[RGB_BRANCH], branches[PCA_BRANCH], weight=candidate.consistency_weight
            )
            return prediction, branch_loss + penalty
        return prediction, prediction.new_zeros(())
    image = pca if candidate.inference_branch == PCA_BRANCH else rgb
    prediction = state.model(image, spectrum)
    return prediction, prediction.new_zeros(())


def _train_one_epoch(
    state: FoldTrainingState,
    candidate: ScreeningCandidate,
    data: ScreeningData,
    epoch: int,
    device: torch.device,
) -> tuple[float, int]:
    state.model.train()
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in state.fold.train_ids], dtype=np.int64)
    generator = torch.Generator().manual_seed(SEED * 100_000 + epoch)
    order = torch.randperm(len(indices), generator=generator).numpy()
    losses = []
    steps = 0
    style = HYPERLEAF_STYLE_CONFIGS[candidate.style_key]
    for batch_number, start in enumerate(range(0, len(order), BATCH_SIZE)):
        selected = indices[order[start : start + BATCH_SIZE]]
        rgb = torch.from_numpy(data.rgb[selected]).to(device=device, dtype=torch.float32)
        pca = torch.from_numpy(data.pca_by_fold[state.fold.fold_id][selected]).to(device=device, dtype=torch.float32)
        spectrum = torch.from_numpy(data.spectrum[selected]).to(device=device, dtype=torch.float32)
        targets_raw = data.targets[selected]
        artifact = data.target_artifacts[state.fold.fold_id]
        mean = np.asarray(artifact.state["mean"], dtype=np.float32)
        scale = np.asarray(artifact.state["scale"], dtype=np.float32)
        normalized = ((targets_raw - mean) / scale).astype(np.float32)
        targets = _target_space(torch.from_numpy(normalized).to(device), candidate)
        augmentation_seed = SEED * 10_000_000 + epoch * 10_000 + batch_number
        rgb, styled_spectrum = apply_hyperleaf_style(rgb, spectrum, config=style, seed=augmentation_seed)
        pca, _ = apply_hyperleaf_style(pca, spectrum, config=style, seed=augmentation_seed)
        state.optimizer.zero_grad(set_to_none=True)
        prediction, auxiliary_loss = _forward(
            state, candidate, rgb, pca, styled_spectrum, targets=targets
        )
        loss = (
            _task_loss(prediction, targets, candidate)
            if candidate.model_key != RGB_PCA_CONSISTENCY_MODEL_KEY
            else auxiliary_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss for {candidate.candidate_id} {state.fold.fold_id}")
        loss.backward()
        gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in state.model.parameters())
        if not gradients_finite:
            raise RuntimeError(f"non-finite gradient for {candidate.candidate_id} {state.fold.fold_id}")
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), GRADIENT_CLIP)
        state.optimizer.step()
        losses.append(float(loss.detach().item()))
        steps += 1
    state.scheduler.step()
    state.completed_steps += steps
    return float(np.mean(losses)), steps


def _predict_fold(
    state: FoldTrainingState,
    candidate: ScreeningCandidate,
    data: ScreeningData,
    device: torch.device,
) -> np.ndarray:
    state.model.eval()
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in state.fold.validation_ids], dtype=np.int64)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), EVALUATION_BATCH_SIZE):
            selected = indices[start : start + EVALUATION_BATCH_SIZE]
            rgb = torch.from_numpy(data.rgb[selected]).to(device=device, dtype=torch.float32)
            pca = torch.from_numpy(data.pca_by_fold[state.fold.fold_id][selected]).to(device=device, dtype=torch.float32)
            spectrum = torch.from_numpy(data.spectrum[selected]).to(device=device, dtype=torch.float32)
            prediction, _ = _forward(state, candidate, rgb, pca, spectrum)
            prediction = prediction.clone()
            prediction[:, 1] = inverse_gsw_target(
                prediction[:, 1], GswTargetTransform(candidate.target_transform)
            )
            outputs.append(prediction.cpu().numpy().astype(np.float64))
    normalized = np.concatenate(outputs)
    artifact = data.target_artifacts[state.fold.fold_id]
    mean = np.asarray(artifact.state["mean"], dtype=np.float64)
    scale = np.asarray(artifact.state["scale"], dtype=np.float64)
    result = normalized * scale + mean
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite prediction for {candidate.candidate_id} {state.fold.fold_id}")
    return result


def _ordered_oof(
    folds: Sequence[ScreeningFold],
    predictions: Mapping[str, np.ndarray],
    data: ScreeningData,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    by_id: dict[str, tuple[str, np.ndarray]] = {}
    for fold in folds:
        values = predictions[fold.fold_id]
        if values.shape != (len(fold.validation_ids), len(HYPERLEAF_TARGETS)):
            raise RuntimeError("fold prediction shape is invalid")
        for sample_id, row in zip(fold.validation_ids, values):
            if sample_id in by_id:
                raise RuntimeError("OOF sample was predicted more than once")
            by_id[sample_id] = (fold.fold_id, row)
    if set(by_id) != set(data.sample_ids):
        raise RuntimeError("OOF prediction coverage is incomplete")
    ids = data.sample_ids
    fold_ids = tuple(by_id[sample_id][0] for sample_id in ids)
    predicted = np.stack([by_id[sample_id][1] for sample_id in ids])
    truth = data.targets.copy()
    return ids, fold_ids, truth, predicted


def _checkpoint_payload(state: FoldTrainingState, candidate: ScreeningCandidate, epoch: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "candidate_id": candidate.candidate_id,
        "candidate_config_hash": candidate.config_hash,
        "fold_id": state.fold.fold_id,
        "seed": SEED,
        "completed_epoch": epoch,
        "completed_training_steps": state.completed_steps,
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
    }


def _save_recovery(candidate_root: Path, state: FoldTrainingState, candidate: ScreeningCandidate, epoch: int) -> None:
    slot = epoch % 2
    _atomic_torch(candidate_root / "folds" / state.fold.fold_id / "recovery" / f"slot-{slot}.pt", _checkpoint_payload(state, candidate, epoch))


def _save_selected(candidate_root: Path, states: Sequence[FoldTrainingState], candidate: ScreeningCandidate, epoch: int) -> None:
    for state in states:
        _atomic_torch(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "selected.pt", _checkpoint_payload(state, candidate, epoch))


def _load_resume(candidate_root: Path, states: Sequence[FoldTrainingState], candidate: ScreeningCandidate) -> tuple[int, list[dict[str, Any]], float]:
    marker = candidate_root / "common-recovery.json"
    if not marker.exists():
        return 0, [], -math.inf
    payload = json.loads(marker.read_text(encoding="utf-8"))
    epoch = int(payload["completed_epoch"])
    if payload["candidate_config_hash"] != candidate.config_hash:
        raise RuntimeError("candidate recovery config hash mismatch")
    for state in states:
        checkpoint = torch.load(
            candidate_root / "folds" / state.fold.fold_id / "recovery" / f"slot-{epoch % 2}.pt",
            map_location=next(state.model.parameters()).device,
            weights_only=False,
        )
        if checkpoint["completed_epoch"] != epoch or checkpoint["candidate_config_hash"] != candidate.config_hash:
            raise RuntimeError("candidate fold recovery boundary mismatch")
        state.model.load_state_dict(checkpoint["model"], strict=True)
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        state.scheduler.load_state_dict(checkpoint["scheduler"])
        state.completed_steps = int(checkpoint["completed_training_steps"])
    return epoch, list(payload.get("history", [])), float(payload.get("best_score", -math.inf))


def _publish_predictions(
    execution_root: Path,
    candidate_root: Path,
    candidate: ScreeningCandidate,
    rung_epoch: int,
    selected_epoch: int,
    folds: Sequence[ScreeningFold],
    data: ScreeningData,
    predictions: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    store = FailureAtomicArtifactStore(OUTPUT_ROOT)
    writer = PredictionArtifactWriter(store)
    references = []
    records_for_hash = []
    for fold in folds:
        indices = np.asarray([data.id_to_index[sample_id] for sample_id in fold.validation_ids])
        records = tuple(
            PredictionRecord(
                key=PredictionKey(
                    sample_id=sample_id,
                    fold_id=fold.fold_id,
                    seed=SEED,
                    inference_branch=candidate.inference_branch,
                ),
                target_name="three-target-output",
                y_true=tuple(float(value) for value in data.targets[index]),
                y_pred=tuple(float(value) for value in prediction),
            )
            for sample_id, index, prediction in zip(fold.validation_ids, indices, predictions[fold.fold_id])
        )
        relative = (candidate_root / "predictions" / f"rung-{rung_epoch:03d}-{fold.fold_id}.canonical.json").relative_to(OUTPUT_ROOT).as_posix()
        published = writer.publish_shards(
            artifact_id=f"task-14-12-{candidate.safe_id}-rung-{rung_epoch:03d}-{fold.fold_id}",
            run_id=f"screen-{candidate.safe_id}-{fold.fold_id}-seed-42",
            expected_sample_ids=fold.validation_ids,
            shards=(records,),
            relative_path=relative,
        )
        references.append(published.reference.model_dump(mode="json"))
        records_for_hash.extend(record.model_dump(mode="json") for record in records)
    ids, fold_ids, truth, forecast = _ordered_oof(folds, predictions, data)
    metrics = compute_grouped_metrics(ids, fold_ids, truth, forecast)
    oof_payload = {
        "schema_version": "1.0",
        "candidate_id": candidate.candidate_id,
        "candidate_config_hash": candidate.config_hash,
        "seed": SEED,
        "rung_epoch": rung_epoch,
        "selected_epoch": selected_epoch,
        "source_sample_ids": ids,
        "source_fold_ids": fold_ids,
        "fold_prediction_artifacts": references,
        "records": records_for_hash,
        "metrics": metrics,
    }
    oof_payload["source_prediction_hash"] = hash_object(oof_payload)
    _atomic_json(candidate_root / "predictions" / f"rung-{rung_epoch:03d}-exact-inner-oof.json", oof_payload)
    return metrics, str(oof_payload["source_prediction_hash"]), references


def _metric_score(metrics: Mapping[str, Any]) -> float:
    value = metrics["pooled_oof"]["targets"]["Gsw"]["r2"]
    if value is None or not math.isfinite(float(value)):
        return -math.inf
    return float(value)


def _make_states(candidate: ScreeningCandidate, folds: Sequence[ScreeningFold], device: torch.device) -> list[FoldTrainingState]:
    states = []
    for fold in folds:
        model = _build_model(candidate, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RUNG_EPOCHS[-1])
        states.append(FoldTrainingState(fold=fold, model=model, optimizer=optimizer, scheduler=scheduler))
    return states


def _train_candidate_to_rung(
    execution_root: Path,
    candidate: ScreeningCandidate,
    folds: Sequence[ScreeningFold],
    data: ScreeningData,
    target_epoch: int,
    template: Any,
    environment: Any,
    device: torch.device,
    overall_started_ns: int,
) -> dict[str, Any]:
    candidate_root = execution_root / "candidates" / candidate.safe_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(candidate_root / "candidate.json", {**asdict(candidate), "config_hash": candidate.config_hash})
    states = _make_states(candidate, folds, device)
    completed_epoch, history, best_score = _load_resume(candidate_root, states, candidate)
    selected_epoch = int(max(history, key=lambda row: (float(row["score"]), -int(row["epoch"])), default={"epoch": 0})["epoch"])
    candidate_started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    best_predictions: dict[str, np.ndarray] | None = None
    selected_npz = candidate_root / "selected-oof.npz"
    if selected_npz.exists():
        with np.load(selected_npz, allow_pickle=False) as stored:
            best_predictions = {fold.fold_id: np.asarray(stored[fold.fold_id], dtype=np.float64) for fold in folds}

    for epoch in range(completed_epoch + 1, target_epoch + 1):
        fold_losses = {}
        for state in states:
            loss, _ = _train_one_epoch(state, candidate, data, epoch, device)
            fold_losses[state.fold.fold_id] = loss
            _save_recovery(candidate_root, state, candidate, epoch)
        epoch_predictions = {state.fold.fold_id: _predict_fold(state, candidate, data, device) for state in states}
        ids, fold_ids, truth, forecast = _ordered_oof(folds, epoch_predictions, data)
        metrics = compute_grouped_metrics(ids, fold_ids, truth, forecast)
        score = _metric_score(metrics)
        history.append(
            {
                "epoch": epoch,
                "score": score,
                "pooled_r2": {target: metrics["pooled_oof"]["targets"][target]["r2"] for target in HYPERLEAF_TARGETS},
                "pooled_mae": {target: metrics["pooled_oof"]["targets"][target]["mae"] for target in HYPERLEAF_TARGETS},
                "fold_losses": fold_losses,
                "completed_training_steps": sum(state.completed_steps for state in states),
            }
        )
        if score > best_score:
            best_score = score
            selected_epoch = epoch
            best_predictions = epoch_predictions
            _save_selected(candidate_root, states, candidate, epoch)
            stream = io.BytesIO()
            np.savez_compressed(stream, **best_predictions)
            _atomic_bytes(selected_npz, stream.getvalue())
        marker = {
            "schema_version": "1.0",
            "candidate_id": candidate.candidate_id,
            "candidate_config_hash": candidate.config_hash,
            "completed_epoch": epoch,
            "best_score": best_score,
            "selected_epoch": selected_epoch,
            "history": history,
        }
        _atomic_json(candidate_root / "common-recovery.json", marker)
        elapsed_ms = (time.monotonic_ns() - overall_started_ns) // 1_000_000
        measurements = RuntimeMeasurements(
            completed_epochs=epoch,
            completed_training_steps=max(state.completed_steps for state in states),
            wall_clock_duration_ms=int(elapsed_ms),
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
            device_identity=environment.device_identity,
            numeric_precision="float32",
            parameter_count=sum(parameter.numel() for parameter in states[0].model.parameters()),
            input_volume_bytes=data.input_volume_bytes,
        )
        assessment = assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch)
        if assessment.hard_exceeded:
            raise RuntimeError(f"hard screening budget exceeded at epoch {epoch}: {assessment.model_dump(mode='json')}")
        print(json.dumps({"stage": "training", "candidate_id": candidate.candidate_id, "epoch": epoch, "target_epoch": target_epoch, "primary_gsw_r2": score}), flush=True)

    if best_predictions is None or selected_epoch <= 0:
        raise RuntimeError("candidate did not produce a finite selected prediction")
    for state in states:
        _atomic_torch(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "last.pt", _checkpoint_payload(state, candidate, target_epoch))
    metrics, prediction_hash, prediction_refs = _publish_predictions(
        execution_root, candidate_root, candidate, target_epoch, selected_epoch, folds, data, best_predictions
    )
    duration_ms = (time.monotonic_ns() - candidate_started_ns) // 1_000_000
    result = {
        "candidate_id": candidate.candidate_id,
        "safe_id": candidate.safe_id,
        "candidate_config_hash": candidate.config_hash,
        "family": candidate.family,
        "model_key": candidate.model_key,
        "route": candidate.inference_branch,
        "target_routes": candidate.target_routes,
        "style": candidate.style_key,
        "loss": candidate.loss_kind,
        "target_transform": candidate.target_transform,
        "consistency_weight": candidate.consistency_weight,
        "rung_epoch": target_epoch,
        "selected_epoch": selected_epoch,
        "score": _metric_score(metrics),
        "metrics": metrics,
        "source_prediction_hash": prediction_hash,
        "prediction_artifacts": prediction_refs,
        "completed_training_steps_per_fold": {state.fold.fold_id: state.completed_steps for state in states},
        "runtime_ms_this_invocation": int(duration_ms),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "parameter_count": sum(parameter.numel() for parameter in states[0].model.parameters()),
        "selected_checkpoints": {state.fold.fold_id: hash_file(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "selected.pt") for state in states},
        "last_checkpoints": {state.fold.fold_id: hash_file(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "last.pt") for state in states},
        "recovery_checkpoints": {state.fold.fold_id: hash_file(candidate_root / "folds" / state.fold.fold_id / "recovery" / f"slot-{target_epoch % 2}.pt") for state in states},
        "status": "complete",
    }
    _atomic_json(candidate_root / f"rung-{target_epoch:03d}-result.json", result)
    return result


def _negative_record(execution_id: str, candidate: ScreeningCandidate, result: Mapping[str, Any], reason: str, details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "evidence_id": f"{execution_id}-{candidate.safe_id}-{reason}",
        "task_id": "14.12",
        "reason": reason,
        "candidate_id": candidate.candidate_id,
        "tested_configuration": asdict(candidate),
        "candidate_metrics": result.get("metrics", {}),
        "details": dict(details),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _rank(results: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(results, key=lambda item: (-float(item["score"]), str(item["candidate_id"]).encode("utf-8")))


def execute_screening() -> dict[str, Any]:
    if PureWindowsPath(str(WORKSPACE)).drive.casefold() != "g:" or PureWindowsPath(str(DATA_ROOT)).drive.casefold() != "g:":
        raise RuntimeError("task 14.12 is restricted to approved G: roots")
    template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.SCREENING:
        raise RuntimeError("task 14.12 requires the immutable screening template")
    environment = validate_live_environment(template)
    _set_determinism()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    overall_started_ns = time.monotonic_ns()
    execution_id = datetime.now(timezone.utc).strftime("task-14-12-%Y%m%dT%H%M%S%fZ")
    execution_root = OUTPUT_ROOT / "executions" / execution_id
    execution_root.mkdir(parents=True, exist_ok=False)

    readonly_before = {
        "src": _tree_fingerprint(WORKSPACE / "src"),
        "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "train_manifest": hash_file(TRAIN_MANIFEST),
        "train_csv": hash_file(DATA_ROOT / "train.csv"),
    }
    manifest = _load_train_manifest()
    sample_ids, varieties, targets = _load_training_table(manifest)
    folds = build_inner_lovo_folds(sample_ids, varieties)
    protocol_payload = {
        "schema_version": "1.0",
        "protocol_key": PROTOCOL_KEY,
        "dataset_hash": manifest.dataset_hash,
        "seed_set": [SEED],
        "folds": [asdict(fold) for fold in folds],
        "selection": {
            "objective": "pooled_oof.r2.Gsw",
            "direction": MetricDirection.MAXIMIZE.value,
            "tie_break": "candidate_id_utf8_ascending",
            "epoch_tie_break": "earliest_epoch",
            "source": "HyperLeaf train partition inner-variety LOVO only",
        },
        "official_evaluation_budget": {"maximum_for_task": 0, "consumed": 0},
    }
    protocol_hash = hash_object(protocol_payload)
    candidates = build_screening_candidates()
    plan = {
        "schema_version": "1.0",
        "task_id": "14.12",
        "execution_id": execution_id,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_hash": manifest.dataset_hash,
        "protocol": {**protocol_payload, "protocol_hash": protocol_hash},
        "candidate_set": [{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates],
        "candidate_set_hash": hash_object([asdict(candidate) for candidate in candidates]),
        "successive_halving": {
            "fully_predeclared": True,
            "rung_epochs": list(RUNG_EPOCHS),
            "survivors": list(RUNG_SURVIVORS),
            "all_candidates_receive_identical_initial_evidence": True,
            "folds_per_candidate_per_rung": len(folds),
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "numeric_precision": "float32",
            "deterministic": True,
            "device": "cuda:0",
            "process_count": 1,
        },
        "template": {
            "path": "configs/fast_lane/screening.json",
            "content_hash": hash_file(TEMPLATE_PATH),
            "template_hash": template.template_hash,
            "resource_budget": template.resource_budget.model_dump(mode="json"),
        },
        "scope": {
            "read_partition": "train-only",
            "official_test_images_allowed": False,
            "official_labels_allowed": False,
            "solution_csv_allowed": False,
            "official_metric_selection_allowed": False,
            "write_root": "runs_v2/screening/hyperleaf",
        },
    }
    _atomic_json(execution_root / "frozen-screening-plan.json", plan)
    _atomic_json(execution_root / "environment-lock.json", environment.model_dump(mode="json"))
    _atomic_json(execution_root / "protocol.json", {**protocol_payload, "protocol_hash": protocol_hash})

    data = _load_representations(manifest, sample_ids, folds, targets, protocol_hash, execution_root)
    active = list(candidates)
    all_results: dict[str, dict[str, Any]] = {}
    rung_tables = []
    negative_path = execution_root / "negative-evidence.jsonl"
    failures = []
    for rung_epoch, survivor_count in zip(RUNG_EPOCHS, RUNG_SURVIVORS):
        rung_results = []
        for candidate in active:
            try:
                result = _train_candidate_to_rung(
                    execution_root,
                    candidate,
                    folds,
                    data,
                    rung_epoch,
                    template,
                    environment,
                    device,
                    overall_started_ns,
                )
                all_results[candidate.candidate_id] = result
                rung_results.append(result)
            except Exception as exc:
                failure = {
                    "candidate_id": candidate.candidate_id,
                    "rung_epoch": rung_epoch,
                    "error": repr(exc),
                }
                failures.append(failure)
                _append_jsonl(
                    negative_path,
                    _negative_record(execution_id, candidate, {}, "failed_or_unstable", failure),
                )
                print(json.dumps({"stage": "candidate_failed", **failure}), file=sys.stderr, flush=True)
        ranked = _rank(rung_results)
        if not ranked:
            raise RuntimeError(f"all candidates failed at rung {rung_epoch}")
        promoted_ids = {str(item["candidate_id"]) for item in ranked[:survivor_count]}
        for item in ranked[survivor_count:]:
            candidate = next(value for value in active if value.candidate_id == item["candidate_id"])
            _append_jsonl(
                negative_path,
                _negative_record(
                    execution_id,
                    candidate,
                    item,
                    "underperformance",
                    {
                        "rung_epoch": rung_epoch,
                        "primary_objective": "pooled_oof.r2.Gsw",
                        "rank": ranked.index(item) + 1,
                        "promoted_count": survivor_count,
                        "best_promoted_score": float(ranked[0]["score"]),
                        "candidate_score": float(item["score"]),
                    },
                ),
            )
        rung_table = {
            "rung_epoch": rung_epoch,
            "survivor_count": survivor_count,
            "ranked_candidates": [
                {
                    "rank": rank,
                    "candidate_id": item["candidate_id"],
                    "selected_epoch": item["selected_epoch"],
                    "score": item["score"],
                    "Gsw": item["metrics"]["pooled_oof"]["targets"]["Gsw"],
                    "GrainWeight": item["metrics"]["pooled_oof"]["targets"]["GrainWeight"],
                    "PhiPS2": item["metrics"]["pooled_oof"]["targets"]["PhiPS2"],
                    "promoted": item["candidate_id"] in promoted_ids,
                }
                for rank, item in enumerate(ranked, 1)
            ],
        }
        rung_tables.append(rung_table)
        _atomic_json(execution_root / f"rung-{rung_epoch:03d}-candidate-table.json", rung_table)
        active = [candidate for candidate in active if candidate.candidate_id in promoted_ids]

    final_ranked = _rank(all_results[candidate.candidate_id] for candidate in active)
    winner_result = dict(final_ranked[0])
    winner = next(candidate for candidate in candidates if candidate.candidate_id == winner_result["candidate_id"])
    source_ids = tuple(sample_ids)
    frozen_selection = {
        "schema_version": "1.0",
        "decision_id": f"{execution_id}-gsw-selection",
        "candidate_set": [{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates],
        "candidate_scores": {candidate_id: result["score"] for candidate_id, result in sorted(all_results.items())},
        "candidate_status": {
            candidate.candidate_id: (
                "winner" if candidate.candidate_id == winner.candidate_id else
                "completed_dominated_or_pruned" if candidate.candidate_id in all_results else
                "failed_or_unstable"
            )
            for candidate in candidates
        },
        "source_sample_ids": source_ids,
        "source_prediction_hash": winner_result["source_prediction_hash"],
        "metric_direction": "maximize",
        "objective_metric": "pooled_oof.r2.Gsw",
        "tie_break_rule": "candidate_id_utf8_ascending",
        "selected_candidate": winner.candidate_id,
        "frozen_configuration": {
            **asdict(winner),
            "selected_epoch": winner_result["selected_epoch"],
            "preprocessing": "inner-train target normalization and full-cube PCA fitted separately per validation variety",
            "seed": SEED,
            "matched_controls": [candidate.candidate_id for candidate in candidates if candidate.family == "grouped-branch-control"],
        },
        "selection_evidence": "train-partition inner-variety LOVO OOF only",
        "official_test_labels_accessible": False,
        "official_evaluation_budget_consumed": 0,
        "frozen_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    frozen_selection["decision_hash"] = hash_object(frozen_selection)
    _atomic_json(execution_root / "frozen-selection-decision.json", frozen_selection)

    readonly_after = {
        "src": _tree_fingerprint(WORKSPACE / "src"),
        "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "train_manifest": hash_file(TRAIN_MANIFEST),
        "train_csv": hash_file(DATA_ROOT / "train.csv"),
    }
    readonly_unchanged = readonly_before == readonly_after
    if not readonly_unchanged:
        raise RuntimeError("a read-only source, legacy run, manifest, or training table changed")
    elapsed_ms = (time.monotonic_ns() - overall_started_ns) // 1_000_000
    report = {
        "schema_version": "1.0",
        "task_id": "14.12",
        "status": "pass" if not failures else "completed_with_negative_evidence",
        "execution_id": execution_id,
        "command": "conda run --no-capture-output -n TCFM python -m benchmark.runners.hyperleaf_screening --execute",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "dataset_key": "hyperleaf2024",
            "partition": "train-only",
            "official_test_images_read": False,
            "official_test_labels_read": False,
            "solution_csv_read": False,
            "official_metrics_used_for_selection": False,
            "official_evaluation_budget_consumed": 0,
            "strict_multiseed_started": False,
            "official_evaluation_started": False,
            "single_process": True,
            "read_only_inputs_unchanged": readonly_unchanged,
        },
        "environment": {
            **environment.model_dump(mode="json"),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "device_name": torch.cuda.get_device_name(0),
            "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "numeric_precision": "float32",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "identity": {
            "dataset_hash": manifest.dataset_hash,
            "train_manifest_hash": hash_file(TRAIN_MANIFEST),
            "train_csv_hash": data.train_csv_hash,
            "protocol_key": PROTOCOL_KEY,
            "protocol_hash": protocol_hash,
            "candidate_set_hash": plan["candidate_set_hash"],
            "template_hash": template.template_hash,
            "environment_lock_hash": environment.lock_hash,
            "runner_hash": hash_file(Path(__file__)),
            "model_source_hash": hash_file(WORKSPACE / "benchmark" / "models" / "hyperleaf_candidates.py"),
            "transform_source_hash": hash_file(WORKSPACE / "benchmark" / "transforms" / "hyperleaf_candidates.py"),
        },
        "budget": {
            "frozen_limits": template.resource_budget.model_dump(mode="json"),
            "successive_halving": plan["successive_halving"],
            "observed_wall_clock_duration_ms": int(elapsed_ms),
            "observed_peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "input_volume_bytes": data.input_volume_bytes,
        },
        "folds": [
            {
                "fold_id": fold.fold_id,
                "validation_variety": fold.validation_variety,
                "train_count": len(fold.train_ids),
                "validation_count": len(fold.validation_ids),
            }
            for fold in folds
        ],
        "rung_candidate_tables": rung_tables,
        "winner": winner_result,
        "frozen_selection": frozen_selection,
        "failures": failures,
        "negative_evidence_path": "negative-evidence.jsonl",
        "artifacts": {
            "frozen_plan": "frozen-screening-plan.json",
            "protocol": "protocol.json",
            "environment_lock": "environment-lock.json",
            "frozen_selection": "frozen-selection-decision.json",
            "candidate_root": "candidates/",
        },
        "caveats": [
            "This is single-seed candidate-screening evidence and is not strict multi-seed or official evidence.",
            "Successive halving was frozen before training: all candidates received four complete variety folds through epoch 20; only promoted candidates continued.",
            "The screening objective is pooled train-partition inner-LOVO Gsw R2. GrainWeight and PhiPS2 are retained as regression guards but do not determine the winner.",
            "No official test image, official label, solution.csv value, or official metric was read or used.",
        ],
    }
    _atomic_json(execution_root / "task-14.12-report.json", report)
    summary = {
        "status": report["status"],
        "execution_id": execution_id,
        "report": (execution_root / "task-14.12-report.json").relative_to(WORKSPACE).as_posix(),
        "selected_candidate": winner.candidate_id,
        "selected_epoch": winner_result["selected_epoch"],
        "primary_gsw_r2": winner_result["score"],
        "official_evaluation_budget_consumed": 0,
    }
    _atomic_json(OUTPUT_ROOT / "latest-task-14.12-summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run complete governed task-14.12 screening")
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        parser.error("--execute is required")
    try:
        result = execute_screening()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
