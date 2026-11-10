"""Leakage-safe task-24.2 HyperLeaf nested LOVO remediation.

This runner creates one independent selection decision per held-out variety.
Candidate, route, and epoch selection use only complete inner-variety OOF
predictions from that outer fold.  The decision and fold-scoped preprocessing
are frozen before any outer metric is computed.  Task-14.12/14.13 evidence is
read only for immutable candidate declarations and supersession lineage; its
scores never enter selection.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS, HYPERLEAF_VARIETIES
from benchmark.catalog.raw_cube import open_tiff_cube
from benchmark.metrics import PredictionArtifactWriter
from benchmark.models.hyperleaf_candidates import create_hyperleaf_candidate_registry
from benchmark.protocols.hyperleaf_lovo import plan_hyperleaf_lovo
from benchmark.protocols.planner import ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.registry.records import FrozenRunIdentity
from benchmark.remediation import (
    RemediationPlanningContext,
    RemediationProfile,
    load_remediation_template,
)
from benchmark.reporting import ClaimEvidence, ClaimGate
from benchmark.schema import (
    EnvironmentLock,
    ExecutionMode,
    FitBoundary,
    FitBoundaryScope,
    ImplementationKind,
    MetricDirection,
    NumericPrecision,
    PredictionKey,
    PredictionRecord,
    RuntimeMeasurements,
    SampleManifest,
    assess_resource_budget,
    canonical_serialize,
    capture_environment_lock,
    hash_file,
    hash_object,
)
from benchmark.transforms.hyperleaf_candidates import HYPERLEAF_STYLE_CONFIGS, apply_hyperleaf_style
from benchmark.transforms.preprocessing import PreprocessingManager

from .hyperleaf_screening import (
    BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    TARGET_ROUTES,
    TRAIN_MANIFEST,
    WEIGHT_DECAY,
    ScreeningCandidate,
    ScreeningData,
    ScreeningFold,
    _atomic_bytes,
    _atomic_json,
    _atomic_torch,
    _forward,
    _load_train_manifest,
    _load_training_table,
    _predict_fold,
    _target_space,
    _tree_fingerprint,
    build_screening_candidates,
    compute_grouped_metrics,
)
from .hyperleaf_strict import aggregate_candidate_metrics

WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
PROFILE_PATH = WORKSPACE / "configs" / "remediation" / "v1" / "deterministic-float32.json"
PROFILE_ROOT = WORKSPACE / "runs_v2" / "remediation" / "v1" / "deterministic-float32"
TASK_ROOT = PROFILE_ROOT / "hyperleaf-nested"
ARTIFACT_ROOT = WORKSPACE / "artifacts" / "remediation" / "v1" / "deterministic-float32" / "hyperleaf-nested"
LEGACY_SCREENING_SUMMARY = WORKSPACE / "runs_v2" / "screening" / "hyperleaf" / "latest-task-14.12-summary.json"
LEGACY_STRICT_SUMMARY = WORKSPACE / "runs_v2" / "strict" / "hyperleaf" / "latest-task-14.13-summary.json"
SEEDS = (7, 42, 123)
MAX_SELECTION_EPOCH = 80
INSTABILITY_STD_THRESHOLD = 0.05
COMMAND_EXECUTE = "conda run --no-capture-output -n TCFM python -m benchmark.runners.hyperleaf_nested_remediation --execute"
REQUIRED_CANDIDATE_IDS = (
    "branch::independent-target",
    "branch::rgb-pca-consistency",
    "branch::style-augmentation",
    "branch::train-fold-pca",
    "gsw::rgb-pca-consistency-0.05::mse::standardized",
)
LEGACY_WINNER_ID = "branch::rgb-pca-consistency"


@dataclass(frozen=True)
class OuterContext:
    fold: ScreeningFold
    inner_folds: tuple[ScreeningFold, ...]


@dataclass
class TrainState:
    fold: ScreeningFold
    seed: int
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    completed_steps: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _publish_json_once(path: Path, value: Any) -> None:
    content = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    _atomic_bytes(path, content)


def _append_once(path: Path, value: Mapping[str, Any], *, key: str = "evidence_id") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    identifier = str(value[key])
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and str(json.loads(line).get(key, "")) == identifier:
                return
    with path.open("ab") as stream:
        stream.write(_json_bytes(dict(value)))
        stream.flush()
        os.fsync(stream.fileno())


def build_remediation_candidates(
    declarations: Sequence[ScreeningCandidate] | None = None,
) -> tuple[ScreeningCandidate, ...]:
    """Resolve only the five predeclared current candidate/control identities."""

    by_id = {item.candidate_id: item for item in (declarations or build_screening_candidates())}
    missing = [candidate_id for candidate_id in REQUIRED_CANDIDATE_IDS if candidate_id not in by_id]
    if missing:
        raise RuntimeError("immutable task-24.2 candidate declarations are missing: " + ", ".join(missing))
    selected = tuple(by_id[candidate_id] for candidate_id in REQUIRED_CANDIDATE_IDS)
    expected = {
        "branch::rgb-pca-consistency": ("target-routed", 0.10, "mse", "standardized"),
        "gsw::rgb-pca-consistency-0.05::mse::standardized": ("rgb-pca-average", 0.05, "mse", "standardized"),
        "branch::train-fold-pca": ("pca", 0.0, "mse", "standardized"),
        "branch::style-augmentation": ("rgb", 0.0, "mse", "standardized"),
        "branch::independent-target": ("rgb", 0.0, "mse", "standardized"),
    }
    for candidate in selected:
        observed = (candidate.inference_branch, candidate.consistency_weight, candidate.loss_kind, candidate.target_transform)
        if observed != expected[candidate.candidate_id]:
            raise RuntimeError(f"immutable candidate declaration drift: {candidate.candidate_id}")
    if selected[1].target_routes != TARGET_ROUTES:
        raise RuntimeError("target-routed winner declaration drift")
    return selected


def build_nested_contexts(protocol: Any, manifest: SampleManifest) -> tuple[OuterContext, ...]:
    """Build four outer contexts and validate exact three-variety inner OOF coverage."""

    groups = {record.sample_id: str(record.group_id) for record in manifest.records}
    contexts: list[OuterContext] = []
    all_outer_ids: list[str] = []
    for outer in protocol.folds:
        held_groups = {groups[sample_id] for sample_id in outer.outer_test_ids}
        if len(held_groups) != 1:
            raise RuntimeError(f"outer fold does not hold exactly one variety: {outer.fold_id}")
        held = next(iter(held_groups))
        outer_fold = ScreeningFold(
            fold_id=outer.fold_id,
            validation_variety=held,
            train_ids=outer.outer_train_ids,
            validation_ids=outer.outer_test_ids,
        )
        inner_folds: list[ScreeningFold] = []
        observed_inner: list[str] = []
        for inner in outer.inner_folds:
            validation_groups = {groups[sample_id] for sample_id in inner.validation_ids}
            train_groups = {groups[sample_id] for sample_id in inner.train_ids}
            if len(validation_groups) != 1 or held in validation_groups or held in train_groups:
                raise RuntimeError(f"held-out variety entered inner evidence: {outer.fold_id}/{inner.fold_id}")
            if set(inner.train_ids) & set(inner.validation_ids):
                raise RuntimeError(f"inner train/validation overlap: {inner.fold_id}")
            if not (set(inner.train_ids) | set(inner.validation_ids)).issubset(set(outer.outer_train_ids)):
                raise RuntimeError(f"inner fold leaves outer-train boundary: {inner.fold_id}")
            validation_variety = next(iter(validation_groups))
            inner_folds.append(ScreeningFold(
                fold_id=f"{outer.fold_id}--{inner.fold_id}",
                validation_variety=validation_variety,
                train_ids=inner.train_ids,
                validation_ids=inner.validation_ids,
            ))
            observed_inner.extend(inner.validation_ids)
        if len(inner_folds) != 3:
            raise RuntimeError(f"outer fold does not contain exactly three inner LOVO folds: {outer.fold_id}")
        if len(observed_inner) != len(set(observed_inner)) or set(observed_inner) != set(outer.outer_train_ids):
            raise RuntimeError(f"inner OOF coverage is not exact for {outer.fold_id}")
        contexts.append(OuterContext(outer_fold, tuple(inner_folds)))
        all_outer_ids.extend(outer.outer_test_ids)
    if tuple(context.fold.validation_variety for context in contexts) != HYPERLEAF_VARIETIES:
        raise RuntimeError("outer variety order differs from frozen LOVO protocol")
    if len(all_outer_ids) != len(set(all_outer_ids)) or set(all_outer_ids) != set(manifest.canonical_order):
        raise RuntimeError("outer LOVO coverage is not exact")
    return tuple(contexts)


def select_outer_decision(
    *,
    context: OuterContext,
    candidates: Sequence[ScreeningCandidate],
    candidate_evidence: Mapping[str, Mapping[str, Any]],
    execution_id: str,
) -> dict[str, Any]:
    """Freeze candidate/epoch/route using only one outer context's inner evidence."""

    outer_train = set(context.fold.train_ids)
    outer_test = set(context.fold.validation_ids)
    source_ids = tuple(sorted({sample_id for fold in context.inner_folds for sample_id in fold.validation_ids}, key=str.encode))
    if set(source_ids) != outer_train or set(source_ids) & outer_test:
        raise RuntimeError("selection source is not exact outer-train-only inner OOF")
    if set(candidate_evidence) != {candidate.candidate_id for candidate in candidates}:
        raise RuntimeError("selection evidence does not cover the complete candidate set")
    candidate_scores: dict[str, float] = {}
    candidate_epochs: dict[str, int] = {}
    prediction_hashes: dict[str, Mapping[str, str]] = {}
    for candidate in candidates:
        evidence = candidate_evidence[candidate.candidate_id]
        evidence_ids = set(str(item) for item in evidence["source_sample_ids"])
        if evidence_ids != outer_train or evidence_ids & outer_test:
            raise RuntimeError(f"candidate inner source boundary mismatch: {candidate.candidate_id}")
        score = float(evidence["score"])
        epoch = int(evidence["selected_epoch"])
        if not math.isfinite(score) or not 1 <= epoch <= MAX_SELECTION_EPOCH:
            raise RuntimeError(f"invalid candidate selection evidence: {candidate.candidate_id}")
        candidate_scores[candidate.candidate_id] = score
        candidate_epochs[candidate.candidate_id] = epoch
        prediction_hashes[candidate.candidate_id] = dict(evidence["source_prediction_hashes"])
    winner = min(candidate_scores, key=lambda key: (-candidate_scores[key], key.encode("utf-8")))
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    decision = {
        "schema_version": "1.0",
        "task_id": "24.2",
        "decision_id": f"{execution_id}-{context.fold.fold_id}-selection",
        "fold_id": context.fold.fold_id,
        "held_out_variety": context.fold.validation_variety,
        "candidate_set": [{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates],
        "candidate_scores": candidate_scores,
        "candidate_selected_epochs": candidate_epochs,
        "candidate_source_prediction_hashes": prediction_hashes,
        "source_sample_ids": source_ids,
        "source_sample_ids_hash": hash_object(source_ids),
        "source_outer_train_count": len(source_ids),
        "outer_test_overlap_count": 0,
        "metric_direction": MetricDirection.MAXIMIZE.value,
        "objective_metric": "mean_over_seeds.inner_pooled_oof.r2.Gsw",
        "tie_break_rule": "candidate_id_utf8_ascending",
        "epoch_tie_break_rule": "earliest_epoch",
        "selected_candidate": winner,
        "selected_epoch": candidate_epochs[winner],
        "selected_route": by_id[winner].inference_branch,
        "selected_candidate_config_hash": by_id[winner].config_hash,
        "seeds_used_for_selection": list(SEEDS),
        "outer_metrics_available_at_freeze": False,
        "legacy_task_14_12_scores_used": False,
        "legacy_task_14_13_metrics_used": False,
        "frozen_utc": _utc_now(),
    }
    decision["decision_hash"] = hash_object(decision)
    return decision


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _training_sources_identity(manifest: SampleManifest) -> str:
    """Hash only approved training references without enumerating official files."""

    return hash_object({
        "train_manifest_hash": hash_file(TRAIN_MANIFEST),
        "train_csv_hash": hash_file(DATA_ROOT / "train.csv"),
        "training_inputs": [
            {
                "sample_id": record.sample_id,
                "references": dict(record.input_references),
                "content_hashes": dict(record.input_content_hashes),
            }
            for record in manifest.records
        ],
    })


def _planning_context(template: Any, environment: EnvironmentLock) -> RemediationPlanningContext:
    return RemediationPlanningContext(
        conda_environment=os.environ.get("CONDA_DEFAULT_ENV", ""),
        environment_lock=environment,
        device="cuda:0",
        process_count=1,
        numeric_precision=NumericPrecision.FLOAT32,
        execution_mode=ExecutionMode.DETERMINISTIC,
        autocast_policy=template.autocast_policy,
        gradient_accumulation=template.gradient_accumulation,
    )


def _validate_environment(template: Any) -> EnvironmentLock:
    environment = capture_environment_lock()
    template.validate_workspace_root(WORKSPACE)
    template.validate_planning_context(_planning_context(template, environment))
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("task 24.2 requires the local RTX 5080")
    return environment


def _fit_all_preprocessing(
    contexts: Sequence[OuterContext],
    sample_ids: Sequence[str],
    targets: np.ndarray,
    raw_spectra: np.ndarray,
    dataset_hash: str,
    protocol_hash: str,
    execution_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, FitBoundary]]:
    manager = PreprocessingManager()
    index = {sample_id: position for position, sample_id in enumerate(sample_ids)}
    target_artifacts: dict[str, Any] = {}
    pca_artifacts: dict[str, Any] = {}
    boundaries: dict[str, FitBoundary] = {}
    for context in contexts:
        folds = (*context.inner_folds, context.fold)
        for fold in folds:
            is_outer = fold.fold_id == context.fold.fold_id
            boundary = FitBoundary(
                fold_id=context.fold.fold_id,
                scope=FitBoundaryScope.OUTER_TRAIN if is_outer else FitBoundaryScope.INNER_TRAIN,
                inner_fold_id=None if is_outer else fold.fold_id,
                allowed_fit_sample_ids=fold.train_ids,
                outer_test_sample_ids=context.fold.validation_ids,
            )
            train_indices = np.asarray([index[sample_id] for sample_id in fold.train_ids], dtype=np.int64)
            target = manager.fit_target_normalization(
                targets[train_indices], fold.train_ids, boundary,
                dataset_hash=dataset_hash, protocol_hash=protocol_hash,
            )
            pca = manager.fit_pca(
                raw_spectra[train_indices], fold.train_ids, boundary,
                dataset_hash=dataset_hash, protocol_hash=protocol_hash, n_components=3, whiten=False,
            )
            if set(target.fit_sample_ids) != set(fold.train_ids) or set(pca.fit_sample_ids) != set(fold.train_ids):
                raise RuntimeError(f"preprocessing fit IDs differ from allowed train IDs: {fold.fold_id}")
            if set(target.fit_sample_ids) & set(context.fold.validation_ids) or set(pca.fit_sample_ids) & set(context.fold.validation_ids):
                raise RuntimeError(f"outer variety entered fitted preprocessing: {fold.fold_id}")
            target_artifacts[fold.fold_id] = target
            pca_artifacts[fold.fold_id] = pca
            boundaries[fold.fold_id] = boundary
            _publish_json_once(execution_root / "transforms" / fold.fold_id / "target-normalization.json", target.model_dump(mode="json"))
            _publish_json_once(execution_root / "transforms" / fold.fold_id / "pca.json", pca.model_dump(mode="json"))
    return target_artifacts, pca_artifacts, boundaries


def _resize_chw(values: np.ndarray, size: tuple[int, int] = (32, 64)) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False).squeeze(0).numpy().astype(np.float32, copy=False)


def _load_nested_data(
    manifest: SampleManifest,
    sample_ids: Sequence[str],
    contexts: Sequence[OuterContext],
    targets: np.ndarray,
    protocol_hash: str,
    execution_root: Path,
) -> ScreeningData:
    records = {record.sample_id: record for record in manifest.records}
    raw_spectra: list[np.ndarray] = []
    actual_bytes_read = (DATA_ROOT / "train.csv").stat().st_size
    for sample_id in sample_ids:
        path = DATA_ROOT / Path(records[sample_id].input_references["full_cube"])
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError("training cube left approved G: data root")
        actual_bytes_read += path.stat().st_size
        with open_tiff_cube(path, axis_order="BHW") as reader:
            raw_spectra.append(np.asarray(reader.memmap_view(), dtype=np.float64).mean(axis=(1, 2)))
    raw_spectrum_array = np.stack(raw_spectra)
    target_artifacts, pca_artifacts, boundaries = _fit_all_preprocessing(
        contexts, sample_ids, targets, raw_spectrum_array, manifest.dataset_hash, protocol_hash, execution_root
    )
    rgb_rows: list[np.ndarray] = []
    spectrum_rows: list[np.ndarray] = []
    pca_rows: dict[str, list[np.ndarray]] = {fold_id: [] for fold_id in pca_artifacts}
    manager = PreprocessingManager()
    for position, sample_id in enumerate(sample_ids):
        path = DATA_ROOT / Path(records[sample_id].input_references["full_cube"])
        actual_bytes_read += path.stat().st_size
        with open_tiff_cube(path, axis_order="BHW") as reader:
            cube = reader.memmap_view()
            bands = reader.shape[0]
            band_indices = tuple(min(bands - 1, int(bands * ratio)) for ratio in (0.72, 0.48, 0.20))
            rgb = np.stack([np.asarray(cube[index], dtype=np.float32) for index in band_indices])
            rgb = rgb / (65535.0 if float(np.nanmax(rgb)) > 1.0 else 1.0)
            image = Image.fromarray(np.moveaxis((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 0, -1))
            resized = image.resize((64, 32), Image.Resampling.BILINEAR)
            rgb_rows.append(np.moveaxis(np.asarray(resized, dtype=np.float32) / 255.0, -1, 0))
            raw = raw_spectrum_array[position]
            interpolated = np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, raw.size), raw)
            scale = float(interpolated.std())
            spectrum_rows.append(((interpolated - interpolated.mean()) / (scale if scale > 1e-12 else 1.0)).astype(np.float32))
            pixels = np.moveaxis(np.asarray(cube, dtype=np.float32), 0, -1).reshape(-1, bands)
            for fold_id, artifact in pca_artifacts.items():
                projected = manager.apply(artifact, pixels, boundaries[fold_id]).reshape(reader.shape[1], reader.shape[2], 3)
                minimum = projected.min((0, 1), keepdims=True)
                maximum = projected.max((0, 1), keepdims=True)
                denominator = np.where(maximum - minimum > 1e-12, maximum - minimum, 1.0)
                pca_rows[fold_id].append(_resize_chw(np.moveaxis(((projected - minimum) / denominator).astype(np.float32), -1, 0)))
        if (position + 1) % 100 == 0 or position + 1 == len(sample_ids):
            print(json.dumps({"stage": "task_24_2_representation_materialization", "completed": position + 1, "total": len(sample_ids)}), flush=True)
    rgb_array = np.stack(rgb_rows).astype(np.float32)
    spectrum_array = np.stack(spectrum_rows).astype(np.float32)
    pca_arrays = {fold_id: np.stack(rows).astype(np.float32) for fold_id, rows in pca_rows.items()}
    if rgb_array.shape != (1590, 3, 32, 64) or spectrum_array.shape != (1590, 128):
        raise RuntimeError("nested representation shape is invalid")
    if any(array.shape != (1590, 3, 32, 64) for array in pca_arrays.values()):
        raise RuntimeError("nested PCA representation shape is invalid")
    return ScreeningData(
        sample_ids=tuple(sample_ids), id_to_index={sample_id: i for i, sample_id in enumerate(sample_ids)},
        varieties=tuple(str(records[sample_id].group_id) for sample_id in sample_ids), targets=targets,
        rgb=rgb_array, spectrum=spectrum_array, pca_by_fold=pca_arrays,
        target_artifacts=target_artifacts, pca_artifacts=pca_artifacts,
        input_volume_bytes=actual_bytes_read, train_csv_hash=hash_file(DATA_ROOT / "train.csv"),
    )


def _replace_adaptive_pool(module: nn.Module) -> int:
    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool1d) and child.output_size == 8:
            setattr(module, name, nn.AvgPool1d(kernel_size=2, stride=2))
            replaced += 1
        else:
            replaced += _replace_adaptive_pool(child)
    return replaced


def _build_state(candidate: ScreeningCandidate, fold: ScreeningFold, seed: int, max_epoch: int, device: torch.device) -> TrainState:
    _set_determinism(seed)
    model = create_hyperleaf_candidate_registry().build(candidate.model_key)
    _replace_adaptive_pool(model)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    return TrainState(fold=fold, seed=seed, model=model, optimizer=optimizer, scheduler=scheduler)


def _task_loss(prediction: torch.Tensor, targets: torch.Tensor, candidate: ScreeningCandidate) -> torch.Tensor:
    from benchmark.transforms.hyperleaf_candidates import GswLossKind, gsw_regression_loss
    other = (F.mse_loss(prediction[:, 0], targets[:, 0]) + F.mse_loss(prediction[:, 2], targets[:, 2])) / 2.0
    gsw = gsw_regression_loss(prediction[:, 1], targets[:, 1], loss_kind=GswLossKind(candidate.loss_kind))
    return (2.0 * other + gsw) / 3.0


def _train_epoch(state: TrainState, candidate: ScreeningCandidate, data: ScreeningData, epoch: int, device: torch.device) -> float:
    state.model.train()
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in state.fold.train_ids], dtype=np.int64)
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(state.seed * 100_000 + epoch)).numpy()
    style = HYPERLEAF_STYLE_CONFIGS[candidate.style_key]
    losses: list[float] = []
    for batch_number, start in enumerate(range(0, len(order), BATCH_SIZE)):
        selected = indices[order[start:start + BATCH_SIZE]]
        rgb = torch.from_numpy(data.rgb[selected]).to(device=device, dtype=torch.float32)
        pca = torch.from_numpy(data.pca_by_fold[state.fold.fold_id][selected]).to(device=device, dtype=torch.float32)
        spectrum = torch.from_numpy(data.spectrum[selected]).to(device=device, dtype=torch.float32)
        artifact = data.target_artifacts[state.fold.fold_id]
        mean = np.asarray(artifact.state["mean"], dtype=np.float32)
        scale = np.asarray(artifact.state["scale"], dtype=np.float32)
        normalized = ((data.targets[selected] - mean) / scale).astype(np.float32)
        targets = _target_space(torch.from_numpy(normalized).to(device), candidate)
        augmentation_seed = state.seed * 10_000_000 + epoch * 10_000 + batch_number
        rgb, styled_spectrum = apply_hyperleaf_style(rgb, spectrum, config=style, seed=augmentation_seed)
        pca, _ = apply_hyperleaf_style(pca, spectrum, config=style, seed=augmentation_seed)
        state.optimizer.zero_grad(set_to_none=True)
        prediction, auxiliary_loss = _forward(state, candidate, rgb, pca, styled_spectrum, targets=targets)
        loss = auxiliary_loss if candidate.model_key.endswith("rgb-pca-consistency.v1") else _task_loss(prediction, targets, candidate)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite nested training loss")
        loss.backward()
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in state.model.parameters()):
            raise RuntimeError("non-finite nested training gradient")
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), GRADIENT_CLIP)
        state.optimizer.step()
        state.completed_steps += 1
        losses.append(float(loss.detach().item()))
    state.scheduler.step()
    return float(np.mean(losses))


def _checkpoint_payload(state: TrainState, candidate: ScreeningCandidate, epoch: int, identity_hash: str, stage: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "task_id": "24.2", "stage_name": stage,
        "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
        "fold_id": state.fold.fold_id, "seed": state.seed, "completed_epoch": epoch,
        "completed_training_steps": state.completed_steps, "frozen_identity_hash": identity_hash,
        "model": state.model.state_dict(), "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(), "stage": {"next_epoch": epoch + 1, "phase": stage},
    }


def _profile_job(
    *, template: Any, planning: RemediationPlanningContext, execution_id: str,
    candidate: ScreeningCandidate, fold: ScreeningFold, seed: int, data: ScreeningData,
    protocol: Any, parameters: Mapping[str, Any], stage: str,
) -> tuple[Any, Path, Path, str]:
    job_name = f"{execution_id}-{stage}-{candidate.safe_id}-{fold.fold_id}-seed-{seed}"
    job = template.freeze_job(
        planning_context=planning, job_name=job_name,
        dataset_hash=protocol.dataset_hash, protocol_hash=protocol.protocol_hash,
        protocol_key=protocol.protocol_key, model_key=candidate.model_key,
        fold_id=fold.fold_id, seed=seed, input_volume_bytes=data.input_volume_bytes,
        selection_metric_direction=MetricDirection.MAXIMIZE,
        parameters={
            "task_id": "24.2", "stage": stage, "candidate_id": candidate.candidate_id,
            "candidate_config_hash": candidate.config_hash, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP, **dict(parameters),
        },
    )
    identity_path = WORKSPACE / Path(job.frozen_identity_relative_path)
    recovery = WORKSPACE / Path(job.recovery_directory)
    identity_payload = job.frozen_identity.model_dump(mode="json")
    if identity_path.exists():
        observed = FrozenRunIdentity.model_validate(_read_json(identity_path))
        job.require_exact_resume_identity(observed)
    else:
        _publish_json_once(identity_path, identity_payload)
    return job, identity_path.parent, recovery, hash_object(identity_payload)


def _load_state_recovery(state: TrainState, candidate: ScreeningCandidate, recovery: Path, identity_hash: str, device: torch.device) -> int:
    marker_path = recovery / "latest.json"
    if not marker_path.exists():
        return 0
    marker = _read_json(marker_path)
    epoch = int(marker["completed_epoch"])
    if marker.get("frozen_identity_hash") != identity_hash:
        raise RuntimeError("recovery marker identity mismatch")
    checkpoint = torch.load(recovery / f"slot-{epoch % 2}.pt", map_location=device, weights_only=False)
    expected = (candidate.candidate_id, candidate.config_hash, state.fold.fold_id, state.seed, epoch, identity_hash)
    observed = (checkpoint["candidate_id"], checkpoint["candidate_config_hash"], checkpoint["fold_id"], checkpoint["seed"], checkpoint["completed_epoch"], checkpoint["frozen_identity_hash"])
    if observed != expected:
        raise RuntimeError("recovery checkpoint identity mismatch")
    state.model.load_state_dict(checkpoint["model"], strict=True)
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.scheduler.load_state_dict(checkpoint["scheduler"])
    state.completed_steps = int(checkpoint["completed_training_steps"])
    return epoch


def _save_state_recovery(state: TrainState, candidate: ScreeningCandidate, recovery: Path, epoch: int, identity_hash: str, stage: str) -> None:
    payload = _checkpoint_payload(state, candidate, epoch, identity_hash, stage)
    _atomic_torch(recovery / f"slot-{epoch % 2}.pt", payload)
    _atomic_json(recovery / "latest.json", {
        "schema_version": "1.0", "frozen_identity_hash": identity_hash,
        "completed_epoch": epoch, "completed_training_steps": state.completed_steps,
        "state_names": ["model", "optimizer", "scheduler", "stage"],
    })


def _inner_seed_predictions(states: Sequence[TrainState], candidate: ScreeningCandidate, data: ScreeningData, device: torch.device) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    by_id: dict[str, tuple[str, np.ndarray]] = {}
    for state in states:
        prediction = _predict_fold(state, candidate, data, device)
        for sample_id, row in zip(state.fold.validation_ids, prediction):
            if sample_id in by_id:
                raise RuntimeError("inner OOF sample predicted more than once")
            by_id[sample_id] = (state.fold.fold_id, row)
    expected = {sample_id for state in states for sample_id in state.fold.validation_ids}
    if set(by_id) != expected:
        raise RuntimeError("inner OOF prediction coverage is incomplete")
    ids = tuple(sorted(expected, key=str.encode))
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in ids], dtype=np.int64)
    return ids, tuple(by_id[sample_id][0] for sample_id in ids), data.targets[indices].copy(), np.stack([by_id[sample_id][1] for sample_id in ids])


def _persist_inner_oof(
    execution_root: Path, context: OuterContext, candidate: ScreeningCandidate, seed: int,
    selected_epoch: int, states: Sequence[TrainState], data: ScreeningData, device: torch.device,
) -> tuple[str, Mapping[str, Any]]:
    ids, fold_ids, truth, forecast = _inner_seed_predictions(states, candidate, data, device)
    metrics = compute_grouped_metrics(ids, fold_ids, truth, forecast)
    records = [
        {
            "key": {"sample_id": sample_id, "fold_id": fold_id, "seed": seed, "inference_branch": candidate.inference_branch},
            "target_name": "three-target-output", "y_true": [float(value) for value in actual],
            "y_pred": [float(value) for value in predicted], "probability": None,
        }
        for sample_id, fold_id, actual, predicted in zip(ids, fold_ids, truth, forecast)
    ]
    payload = {
        "schema_version": "1.0", "task_id": "24.2", "kind": "exact-inner-oof",
        "outer_fold_id": context.fold.fold_id, "held_out_variety": context.fold.validation_variety,
        "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
        "seed": seed, "selected_epoch": selected_epoch, "source_sample_ids": ids,
        "held_out_overlap_count": len(set(ids) & set(context.fold.validation_ids)),
        "records": records, "metrics": metrics,
    }
    payload["source_prediction_hash"] = hash_object(payload)
    path = execution_root / "selection" / context.fold.fold_id / candidate.safe_id / f"seed-{seed}-inner-oof.json"
    _publish_json_once(path, payload)
    return str(payload["source_prediction_hash"]), metrics


def _run_inner_candidate(
    *, execution_id: str, execution_root: Path, context: OuterContext,
    candidate: ScreeningCandidate, data: ScreeningData, protocol: Any, template: Any,
    planning: RemediationPlanningContext, environment: EnvironmentLock, device: torch.device,
) -> dict[str, Any]:
    candidate_root = execution_root / "selection" / context.fold.fold_id / candidate.safe_id
    result_path = candidate_root / "result.json"
    if result_path.exists():
        return _read_json(result_path)
    states: list[TrainState] = []
    job_metadata: dict[tuple[int, str], tuple[Any, Path, Path, str]] = {}
    for seed in SEEDS:
        for fold in context.inner_folds:
            state = _build_state(candidate, fold, seed, MAX_SELECTION_EPOCH, device)
            metadata = _profile_job(
                template=template, planning=planning, execution_id=execution_id,
                candidate=candidate, fold=fold, seed=seed, data=data, protocol=protocol,
                parameters={"outer_fold_id": context.fold.fold_id, "max_epoch": MAX_SELECTION_EPOCH,
                            "selection_scope_hash": hash_object(context.fold.train_ids)},
                stage="inner-selection",
            )
            states.append(state)
            job_metadata[(seed, fold.fold_id)] = metadata
    common_path = candidate_root / "common-recovery.json"
    history: list[dict[str, Any]] = []
    completed_epoch = 0
    best_score = -math.inf
    selected_epoch = 0
    if common_path.exists():
        common = _read_json(common_path)
        if common.get("candidate_config_hash") != candidate.config_hash:
            raise RuntimeError("inner common recovery candidate mismatch")
        completed_epoch = int(common["completed_epoch"])
        history = list(common.get("history", []))
        best_score = float(common.get("best_score", -math.inf))
        selected_epoch = int(common.get("selected_epoch", 0))
        for state in states:
            _, _, recovery, identity_hash = job_metadata[(state.seed, state.fold.fold_id)]
            recovered = _load_state_recovery(state, candidate, recovery, identity_hash, device)
            if recovered != completed_epoch:
                raise RuntimeError("inner recovery states do not share a verified boundary")
    started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    for epoch in range(completed_epoch + 1, MAX_SELECTION_EPOCH + 1):
        losses: dict[str, float] = {}
        for state in states:
            loss = _train_epoch(state, candidate, data, epoch, device)
            losses[f"seed-{state.seed}/{state.fold.fold_id}"] = loss
            _, _, recovery, identity_hash = job_metadata[(state.seed, state.fold.fold_id)]
            _save_state_recovery(state, candidate, recovery, epoch, identity_hash, "inner-selection")
        seed_scores: dict[str, float] = {}
        seed_metrics: dict[str, Any] = {}
        for seed in SEEDS:
            seed_states = [state for state in states if state.seed == seed]
            ids, fold_ids, truth, forecast = _inner_seed_predictions(seed_states, candidate, data, device)
            metrics = compute_grouped_metrics(ids, fold_ids, truth, forecast)
            value = metrics["pooled_oof"]["targets"]["Gsw"]["r2"]
            score = -math.inf if value is None else float(value)
            seed_scores[str(seed)] = score
            seed_metrics[str(seed)] = {
                "Gsw": metrics["pooled_oof"]["targets"]["Gsw"],
                "GrainWeight": metrics["pooled_oof"]["targets"]["GrainWeight"],
                "PhiPS2": metrics["pooled_oof"]["targets"]["PhiPS2"],
            }
        score = float(np.mean([seed_scores[str(seed)] for seed in SEEDS]))
        history.append({"epoch": epoch, "score": score, "seed_scores": seed_scores, "seed_metrics": seed_metrics, "losses": losses})
        if score > best_score:
            best_score = score
            selected_epoch = epoch
            for state in states:
                _, job_root, _, identity_hash = job_metadata[(state.seed, state.fold.fold_id)]
                _atomic_torch(job_root / "checkpoints" / "selected.pt", _checkpoint_payload(state, candidate, epoch, identity_hash, "inner-selection"))
        _atomic_json(common_path, {
            "schema_version": "1.0", "candidate_id": candidate.candidate_id,
            "candidate_config_hash": candidate.config_hash, "completed_epoch": epoch,
            "best_score": best_score, "selected_epoch": selected_epoch, "history": history,
        })
        measurements = RuntimeMeasurements(
            completed_epochs=epoch,
            completed_training_steps=max(state.completed_steps for state in states),
            wall_clock_duration_ms=int((time.monotonic_ns() - started_ns) // 1_000_000),
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
            device_identity=environment.device_identity, numeric_precision="float32",
            parameter_count=sum(parameter.numel() for parameter in states[0].model.parameters()),
            input_volume_bytes=data.input_volume_bytes,
        )
        assessment = assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch)
        if assessment.hard_exceeded:
            raise RuntimeError(f"inner selection hard budget exceeded at epoch {epoch}")
        print(json.dumps({"stage": "task_24_2_inner_selection", "outer_fold_id": context.fold.fold_id,
                          "candidate_id": candidate.candidate_id, "epoch": epoch,
                          "target_epoch": MAX_SELECTION_EPOCH, "mean_seed_gsw_r2": score}), flush=True)
    if selected_epoch <= 0:
        raise RuntimeError("inner selection produced no finite epoch")
    # Reload the selected state before publishing exact selected-epoch OOF evidence.
    for state in states:
        _, job_root, _, identity_hash = job_metadata[(state.seed, state.fold.fold_id)]
        selected = torch.load(job_root / "checkpoints" / "selected.pt", map_location=device, weights_only=False)
        if int(selected["completed_epoch"]) != selected_epoch or selected["frozen_identity_hash"] != identity_hash:
            raise RuntimeError("selected inner checkpoint identity mismatch")
        state.model.load_state_dict(selected["model"], strict=True)
    prediction_hashes: dict[str, str] = {}
    metrics_by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        source_hash, metrics = _persist_inner_oof(
            execution_root, context, candidate, seed, selected_epoch,
            [state for state in states if state.seed == seed], data, device,
        )
        prediction_hashes[str(seed)] = source_hash
        metrics_by_seed[str(seed)] = metrics
    for state in states:
        _, job_root, recovery, identity_hash = job_metadata[(state.seed, state.fold.fold_id)]
        latest = _read_json(recovery / "latest.json")
        checkpoint = torch.load(recovery / f"slot-{MAX_SELECTION_EPOCH % 2}.pt", map_location="cpu", weights_only=False)
        _atomic_torch(job_root / "checkpoints" / "last.pt", checkpoint)
    result = {
        "schema_version": "1.0", "task_id": "24.2", "status": "complete",
        "outer_fold_id": context.fold.fold_id, "held_out_variety": context.fold.validation_variety,
        "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
        "selected_epoch": selected_epoch, "score": best_score,
        "source_sample_ids": tuple(sorted(context.fold.train_ids, key=str.encode)),
        "source_prediction_hashes": prediction_hashes, "per_seed_inner_metrics": metrics_by_seed,
        "history_hash": hash_object(history), "max_selection_epoch": MAX_SELECTION_EPOCH,
        "completed_training_steps": sum(state.completed_steps for state in states),
        "checkpoints": {
            f"seed-{state.seed}/{state.fold.fold_id}": {
                "selected": {
                    "path": (job_metadata[(state.seed, state.fold.fold_id)][1] / "checkpoints" / "selected.pt").relative_to(WORKSPACE).as_posix(),
                    "hash": hash_file(job_metadata[(state.seed, state.fold.fold_id)][1] / "checkpoints" / "selected.pt"),
                },
                "last": {
                    "path": (job_metadata[(state.seed, state.fold.fold_id)][1] / "checkpoints" / "last.pt").relative_to(WORKSPACE).as_posix(),
                    "hash": hash_file(job_metadata[(state.seed, state.fold.fold_id)][1] / "checkpoints" / "last.pt"),
                },
                "recovery": {
                    "path": (job_metadata[(state.seed, state.fold.fold_id)][2] / f"slot-{MAX_SELECTION_EPOCH % 2}.pt").relative_to(WORKSPACE).as_posix(),
                    "hash": hash_file(job_metadata[(state.seed, state.fold.fold_id)][2] / f"slot-{MAX_SELECTION_EPOCH % 2}.pt"),
                },
            }
            for state in states
        },
        "runtime_ms_this_invocation": int((time.monotonic_ns() - started_ns) // 1_000_000),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "parameter_count": sum(parameter.numel() for parameter in states[0].model.parameters()),
    }
    _publish_json_once(result_path, result)
    return result


def _run_outer_job(
    *, execution_id: str, execution_root: Path, context: OuterContext,
    candidate: ScreeningCandidate, selected_epoch: int, decision: Mapping[str, Any], seed: int,
    data: ScreeningData, protocol: Any, template: Any, planning: RemediationPlanningContext,
    environment: EnvironmentLock, device: torch.device,
) -> dict[str, Any]:
    job, job_root, recovery, identity_hash = _profile_job(
        template=template, planning=planning, execution_id=execution_id, candidate=candidate,
        fold=context.fold, seed=seed, data=data, protocol=protocol,
        parameters={"selected_epoch": selected_epoch, "selection_decision_hash": decision["decision_hash"],
                    "outer_preprocessing_hash": hash_object({
                        "target": data.target_artifacts[context.fold.fold_id].state_hash,
                        "pca": data.pca_artifacts[context.fold.fold_id].state_hash,
                    })},
        stage="outer-refit",
    )
    result_path = job_root / "result.json"
    if result_path.exists():
        result = _read_json(result_path)
        if result.get("status") != "complete":
            raise RuntimeError("immutable outer result is not complete")
        return result
    decision_path = execution_root / "decisions" / f"{context.fold.fold_id}.json"
    if not decision_path.exists() or hash_object({k: v for k, v in decision.items() if k != "decision_hash"}) != decision["decision_hash"]:
        raise RuntimeError("outer evaluation requested before a valid frozen selection decision")
    state = _build_state(candidate, context.fold, seed, selected_epoch, device)
    completed_epoch = _load_state_recovery(state, candidate, recovery, identity_hash, device)
    started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    for epoch in range(completed_epoch + 1, selected_epoch + 1):
        loss = _train_epoch(state, candidate, data, epoch, device)
        _save_state_recovery(state, candidate, recovery, epoch, identity_hash, "outer-refit")
        measurements = RuntimeMeasurements(
            completed_epochs=epoch, completed_training_steps=state.completed_steps,
            wall_clock_duration_ms=int((time.monotonic_ns() - started_ns) // 1_000_000),
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
            device_identity=environment.device_identity, numeric_precision="float32",
            parameter_count=sum(parameter.numel() for parameter in state.model.parameters()),
            input_volume_bytes=data.input_volume_bytes,
        )
        if assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch).hard_exceeded:
            raise RuntimeError(f"outer refit hard budget exceeded at epoch {epoch}")
        print(json.dumps({"stage": "task_24_2_outer_refit", "candidate_id": candidate.candidate_id,
                          "seed": seed, "fold_id": context.fold.fold_id, "epoch": epoch,
                          "target_epoch": selected_epoch, "train_loss": loss}), flush=True)
    checkpoint = _checkpoint_payload(state, candidate, selected_epoch, identity_hash, "outer-refit")
    _atomic_torch(job_root / "checkpoints" / "selected.pt", checkpoint)
    _atomic_torch(job_root / "checkpoints" / "last.pt", checkpoint)
    prediction_path = job_root / "predictions.canonical.json"
    if prediction_path.exists():
        artifact = PredictionArtifactWriter.reload(prediction_path.read_bytes())
        commit_path = job_root / "evaluation-commit.json"
        if commit_path.exists():
            commit = _read_json(commit_path)
            if commit.get("prediction_content_hash") != artifact.content_hash or int(commit.get("successful_evaluations", 0)) != 1:
                raise RuntimeError("completed outer evaluation commit does not match prediction")
        else:
            _publish_json_once(commit_path, {
                "schema_version": "1.0", "successful_evaluations": 1,
                "prediction_content_hash": artifact.content_hash,
                "committed_utc": _utc_now(), "resumed_from_completed_prediction": True,
                "gradient_updates_disabled": True, "fitted_state_updates_disabled": True,
            })
    else:
        # This is the only point at which held-out labels are paired with predictions.
        forecast = _predict_fold(state, candidate, data, device)
        indices = np.asarray([data.id_to_index[sample_id] for sample_id in context.fold.validation_ids], dtype=np.int64)
        records = tuple(
            PredictionRecord(
                key=PredictionKey(sample_id=sample_id, fold_id=context.fold.fold_id, seed=seed,
                                  inference_branch=candidate.inference_branch),
                target_name="three-target-output",
                y_true=tuple(float(value) for value in data.targets[index]),
                y_pred=tuple(float(value) for value in predicted),
            )
            for sample_id, index, predicted in zip(context.fold.validation_ids, indices, forecast)
        )
        relative = prediction_path.relative_to(PROFILE_ROOT).as_posix()
        published = PredictionArtifactWriter(FailureAtomicArtifactStore(PROFILE_ROOT)).publish_shards(
            artifact_id=f"{execution_id}-{candidate.safe_id}-{context.fold.fold_id}-seed-{seed}-outer",
            run_id=f"{execution_id}-{candidate.safe_id}-{context.fold.fold_id}-seed-{seed}",
            expected_sample_ids=context.fold.validation_ids, shards=(records,), relative_path=relative,
        )
        artifact = published.artifact
        _publish_json_once(job_root / "evaluation-commit.json", {
            "schema_version": "1.0", "successful_evaluations": 1,
            "prediction_content_hash": artifact.content_hash, "committed_utc": _utc_now(),
            "gradient_updates_disabled": True, "fitted_state_updates_disabled": True,
        })
    by_id = {record.key.sample_id: record for record in artifact.records}
    if set(by_id) != set(context.fold.validation_ids) or len(by_id) != len(context.fold.validation_ids):
        raise RuntimeError("outer prediction exact coverage failed")
    ordered = [by_id[sample_id] for sample_id in context.fold.validation_ids]
    truth = np.asarray([record.y_true for record in ordered], dtype=np.float64)
    forecast = np.asarray([record.y_pred for record in ordered], dtype=np.float64)
    metrics = compute_grouped_metrics(context.fold.validation_ids, (context.fold.fold_id,) * len(ordered), truth, forecast)
    result = {
        "schema_version": "1.0", "task_id": "24.2", "status": "complete",
        "run_id": f"{execution_id}-{candidate.safe_id}-{context.fold.fold_id}-seed-{seed}",
        "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
        "role": "previously_frozen_winner" if candidate.candidate_id == LEGACY_WINNER_ID else "matched_control",
        "fold_id": context.fold.fold_id, "held_out_variety": context.fold.validation_variety,
        "seed": seed, "selected_epoch": selected_epoch,
        "selection_decision_id": decision["decision_id"], "selection_decision_hash": decision["decision_hash"],
        "metrics": metrics,
        "prediction_artifact": {"relative_path": prediction_path.relative_to(PROFILE_ROOT).as_posix(),
                                "content_hash": artifact.content_hash},
        "checkpoints": {
            "selected": {"path": (job_root / "checkpoints" / "selected.pt").relative_to(WORKSPACE).as_posix(),
                         "hash": hash_file(job_root / "checkpoints" / "selected.pt")},
            "last": {"path": (job_root / "checkpoints" / "last.pt").relative_to(WORKSPACE).as_posix(),
                     "hash": hash_file(job_root / "checkpoints" / "last.pt")},
            "recovery": {"path": (recovery / f"slot-{selected_epoch % 2}.pt").relative_to(WORKSPACE).as_posix(),
                         "hash": hash_file(recovery / f"slot-{selected_epoch % 2}.pt")},
        },
        "runtime_measurements": {
            "completed_epochs": selected_epoch, "completed_training_steps": state.completed_steps,
            "wall_clock_duration_ms": int((time.monotonic_ns() - started_ns) // 1_000_000),
            "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "device_identity": environment.device_identity, "numeric_precision": "float32",
            "parameter_count": sum(parameter.numel() for parameter in state.model.parameters()),
            "input_volume_bytes": data.input_volume_bytes,
        },
        "outer_successful_evaluations": 1, "official_evaluation_budget_consumed": 0,
        "completed_utc": _utc_now(),
    }
    _publish_json_once(result_path, result)
    return result


def _load_outer_records(result: Mapping[str, Any]) -> tuple[PredictionRecord, ...]:
    path = PROFILE_ROOT / Path(str(result["prediction_artifact"]["relative_path"]))
    artifact = PredictionArtifactWriter.reload(path.read_bytes())
    if artifact.content_hash != result["prediction_artifact"]["content_hash"]:
        raise RuntimeError("outer prediction hash changed")
    return artifact.records


def _aggregate_results(
    *, execution_root: Path, sample_ids: Sequence[str], contexts: Sequence[OuterContext],
    candidates: Sequence[ScreeningCandidate], decisions: Mapping[str, Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_summaries: dict[str, Any] = {}
    context_by_fold = {context.fold.fold_id: context for context in contexts}
    for candidate in candidates:
        per_seed: dict[int, Mapping[str, Any]] = {}
        seed_artifacts: dict[str, Any] = {}
        candidate_results = [result for result in results if result["candidate_id"] == candidate.candidate_id]
        for seed in SEEDS:
            members = [result for result in candidate_results if int(result["seed"]) == seed]
            if len(members) != len(contexts):
                raise RuntimeError("candidate aggregate lacks complete outer folds")
            records = tuple(record for member in members for record in _load_outer_records(member))
            by_id = {record.key.sample_id: record for record in records}
            if len(by_id) != len(sample_ids) or set(by_id) != set(sample_ids):
                raise RuntimeError("candidate outer OOF coverage is incomplete")
            ordered = tuple(by_id[sample_id] for sample_id in sample_ids)
            truth = np.asarray([record.y_true for record in ordered], dtype=np.float64)
            forecast = np.asarray([record.y_pred for record in ordered], dtype=np.float64)
            fold_ids = tuple(record.key.fold_id for record in ordered)
            metrics = compute_grouped_metrics(sample_ids, fold_ids, truth, forecast)
            metrics["per_target"] = metrics["pooled_oof"]["targets"]
            per_seed[seed] = metrics
            payload = {
                "schema_version": "1.0", "task_id": "24.2", "candidate_id": candidate.candidate_id,
                "candidate_config_hash": candidate.config_hash, "seed": seed, "sample_count": len(sample_ids),
                "selected_epoch_by_outer": {context.fold.fold_id: int(decisions[context.fold.fold_id]["candidate_selected_epochs"][candidate.candidate_id]) for context in contexts},
                "member_run_ids": [member["run_id"] for member in members],
                "prediction_hashes": [member["prediction_artifact"]["content_hash"] for member in members],
                "metrics": metrics, "records": [record.model_dump(mode="json") for record in ordered],
            }
            payload["aggregate_hash"] = hash_object(payload)
            path = execution_root / "aggregates" / candidate.safe_id / f"seed-{seed}-oof.json"
            _publish_json_once(path, payload)
            seed_artifacts[str(seed)] = {"path": path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(path)}
        aggregate = aggregate_candidate_metrics(per_seed)
        candidate_summaries[candidate.candidate_id] = {
            "role": "previously_frozen_winner" if candidate.candidate_id == LEGACY_WINNER_ID else "matched_control",
            "configuration": {**asdict(candidate), "config_hash": candidate.config_hash},
            "selected_epoch_by_outer": {context.fold.fold_id: int(decisions[context.fold.fold_id]["candidate_selected_epochs"][candidate.candidate_id]) for context in contexts},
            "per_seed_metrics": {str(seed): per_seed[seed] for seed in SEEDS},
            "aggregate_metrics": aggregate, "seed_oof_artifacts": seed_artifacts,
            "member_run_ids": [member["run_id"] for member in candidate_results],
            "prediction_hashes": [member["prediction_artifact"]["content_hash"] for member in candidate_results],
        }
    # Nested policy aggregate uses the candidate independently selected in each outer decision.
    policy_per_seed: dict[int, Mapping[str, Any]] = {}
    policy_artifacts: dict[str, Any] = {}
    for seed in SEEDS:
        members = []
        for context in contexts:
            selected = str(decisions[context.fold.fold_id]["selected_candidate"])
            match = [result for result in results if result["candidate_id"] == selected and result["fold_id"] == context.fold.fold_id and int(result["seed"]) == seed]
            if len(match) != 1:
                raise RuntimeError("nested policy aggregate lacks a unique selected outer run")
            members.append(match[0])
        records = tuple(record for member in members for record in _load_outer_records(member))
        by_id = {record.key.sample_id: record for record in records}
        ordered = tuple(by_id[sample_id] for sample_id in sample_ids)
        truth = np.asarray([record.y_true for record in ordered], dtype=np.float64)
        forecast = np.asarray([record.y_pred for record in ordered], dtype=np.float64)
        fold_ids = tuple(record.key.fold_id for record in ordered)
        metrics = compute_grouped_metrics(sample_ids, fold_ids, truth, forecast)
        metrics["per_target"] = metrics["pooled_oof"]["targets"]
        policy_per_seed[seed] = metrics
        payload = {
            "schema_version": "1.0", "task_id": "24.2", "kind": "nested-selection-policy",
            "seed": seed, "member_run_ids": [member["run_id"] for member in members],
            "selected_candidate_by_outer": {member["fold_id"]: member["candidate_id"] for member in members},
            "prediction_hashes": [member["prediction_artifact"]["content_hash"] for member in members],
            "metrics": metrics, "records": [record.model_dump(mode="json") for record in ordered],
        }
        payload["aggregate_hash"] = hash_object(payload)
        path = execution_root / "aggregates" / "nested-policy" / f"seed-{seed}-oof.json"
        _publish_json_once(path, payload)
        policy_artifacts[str(seed)] = {"path": path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(path)}
    policy_summary = {
        "selected_candidate_by_outer": {fold_id: decision["selected_candidate"] for fold_id, decision in decisions.items()},
        "selected_epoch_by_outer": {fold_id: decision["selected_epoch"] for fold_id, decision in decisions.items()},
        "per_seed_metrics": {str(seed): policy_per_seed[seed] for seed in SEEDS},
        "aggregate_metrics": aggregate_candidate_metrics(policy_per_seed),
        "seed_oof_artifacts": policy_artifacts,
        "member_run_ids": [result["run_id"] for result in results if result["candidate_id"] == decisions[result["fold_id"]]["selected_candidate"]],
    }
    return candidate_summaries, policy_summary


def _claim(score: float, *, controls_complete: bool) -> dict[str, Any]:
    decision = ClaimGate.classify(ClaimEvidence(
        claim_id="task-24.2-hyperleaf-nested-remediation",
        implementation_kind=ImplementationKind.IN_REPO, detected_leakage=False,
        model_identity_mismatch=False, remote_sensing_sanity_track=False,
        random_split=False, outer_data_used_for_model_selection=False, single_run=False,
        outer_data_used_for_ensemble_weight_search=False, outer_data_used_for_threshold_search=False,
        strict_protocol_complete=True, provenance_complete=True, seed_set_complete=True,
        folds_complete=True, matched_input_control_required=True,
        matched_input_control_complete=controls_complete, prediction_artifact_complete=True,
        unified_runner_reproduced=True, score=score,
    ))
    return decision.model_dump(mode="json")


def execute_nested(*, resume_execution_id: str | None = None) -> dict[str, Any]:
    if PureWindowsPath(str(WORKSPACE)).drive.casefold() != "g:" or PureWindowsPath(str(DATA_ROOT)).drive.casefold() != "g:":
        raise RuntimeError("task 24.2 is restricted to approved G: roots")
    template = load_remediation_template(PROFILE_PATH)
    if template.profile is not RemediationProfile.DETERMINISTIC_FLOAT32 or tuple(template.seed_set) != SEEDS:
        raise RuntimeError("task 24.2 requires the immutable deterministic-float32 profile")
    environment = _validate_environment(template)
    planning = _planning_context(template, environment)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    _set_determinism(SEEDS[0])
    overall_started_ns = time.monotonic_ns()

    candidates = build_remediation_candidates()
    manifest = _load_train_manifest()
    sample_ids, _, targets = _load_training_table(manifest)
    planner = ProtocolPlanner(FailureAtomicArtifactStore(PROFILE_ROOT))
    protocol = plan_hyperleaf_lovo(planner, manifest, seed_set=SEEDS, publish=False)
    contexts = build_nested_contexts(protocol, manifest)
    if protocol.evaluation_budget.maximum_successful_evaluations != len(contexts) * len(SEEDS):
        raise RuntimeError("LOVO outer evaluation budget differs from exact fold/seed count")

    execution_id = resume_execution_id or datetime.now(timezone.utc).strftime("task-24-2-%Y%m%dT%H%M%S%fZ")
    execution_root = TASK_ROOT / "executions" / execution_id
    report_path = execution_root / "task-24.2-report.json"
    if resume_execution_id:
        if not execution_root.is_dir():
            raise RuntimeError("requested task-24.2 execution does not exist")
        if report_path.exists():
            report = _read_json(report_path)
            return {"status": "complete", "execution_id": execution_id,
                    "report": report_path.relative_to(WORKSPACE).as_posix(),
                    "claim_status": report["claim_and_promotion"]["decision"]["status"]}
    else:
        execution_root.mkdir(parents=True, exist_ok=False)

    legacy_paths = (LEGACY_SCREENING_SUMMARY, LEGACY_STRICT_SUMMARY)
    legacy_reports = []
    for summary_path in legacy_paths:
        summary = _read_json(summary_path)
        source = WORKSPACE / Path(str(summary["report"]))
        legacy_reports.append({
            "summary_path": summary_path.relative_to(WORKSPACE).as_posix(), "summary_hash": hash_file(summary_path),
            "report_path": source.relative_to(WORKSPACE).as_posix(), "report_hash": hash_file(source),
            "preserved_claim_status": "development", "metrics_used_for_task_24_2_selection": False,
        })
    readonly_before = {
        "src": _tree_fingerprint(WORKSPACE / "src"), "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "training_dataset_sources": _training_sources_identity(manifest), "train_manifest": hash_file(TRAIN_MANIFEST),
        "legacy_evidence": hash_object(legacy_reports), "fast_lane_templates": _tree_fingerprint(WORKSPACE / "configs" / "fast_lane"),
        "caches": _tree_fingerprint(WORKSPACE / "caches"), "checkpoints": _tree_fingerprint(WORKSPACE / "checkpoints"),
    }
    plan = {
        "schema_version": "1.0", "task_id": "24.2", "execution_id": execution_id,
        "created_utc": _utc_now() if not resume_execution_id else _read_json(execution_root / "frozen-plan.json")["created_utc"],
        "dataset_hash": manifest.dataset_hash, "protocol_key": protocol.protocol_key,
        "protocol_hash": protocol.protocol_hash, "seed_set": list(SEEDS),
        "candidate_set": [{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates],
        "candidate_set_hash": hash_object([{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates]),
        "selection": {
            "independent_decisions": len(contexts), "inner_folds_per_outer": 3,
            "max_epoch": MAX_SELECTION_EPOCH, "objective": "mean_over_seeds.inner_pooled_oof.r2.Gsw",
            "direction": "maximize", "candidate_tie_break": "candidate_id_utf8_ascending",
            "epoch_tie_break": "earliest_epoch", "legacy_scores_allowed": False,
            "new_outer_metrics_allowed": False,
        },
        "training": {"optimizer": "AdamW", "learning_rate": LEARNING_RATE,
                     "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                     "evaluation_batch_size": EVALUATION_BATCH_SIZE,
                     "gradient_clip": GRADIENT_CLIP},
        "profile": {"path": PROFILE_PATH.relative_to(WORKSPACE).as_posix(),
                    "file_hash": hash_file(PROFILE_PATH), "config_hash": template.config_hash,
                    "profile": template.profile.value, "output_root": template.output_root,
                    "recovery_root": template.recovery_root, "artifact_root": template.artifact_root},
        "legacy_evidence": legacy_reports,
        "scope": {"workspace": "G:/Datasets/HyperCross", "device": "cuda:0", "process_count": 1,
                  "official_test_images_allowed": False, "official_test_labels_allowed": False,
                  "solution_csv_allowed": False, "official_budget": 0},
    }
    plan["plan_hash"] = hash_object(plan)
    if resume_execution_id:
        if _json_bytes(_read_json(execution_root / "frozen-plan.json")) != _json_bytes(plan):
            raise RuntimeError("resume plan differs from frozen task-24.2 identity")
    else:
        _publish_json_once(execution_root / "frozen-plan.json", plan)
        _publish_json_once(execution_root / "environment-lock.json", environment.model_dump(mode="json"))
        _atomic_bytes(execution_root / "protocol.canonical.json", canonical_serialize(protocol))

    data = _load_nested_data(manifest, sample_ids, contexts, targets, protocol.protocol_hash, execution_root)
    decisions: dict[str, Mapping[str, Any]] = {}
    selection_results: dict[str, dict[str, Mapping[str, Any]]] = {}
    for context in contexts:
        evidence: dict[str, Mapping[str, Any]] = {}
        for candidate in candidates:
            result = _run_inner_candidate(
                execution_id=execution_id, execution_root=execution_root, context=context,
                candidate=candidate, data=data, protocol=protocol, template=template,
                planning=planning, environment=environment, device=device,
            )
            evidence[candidate.candidate_id] = result
        decision = select_outer_decision(
            context=context, candidates=candidates, candidate_evidence=evidence, execution_id=execution_id
        )
        _publish_json_once(execution_root / "decisions" / f"{context.fold.fold_id}.json", decision)
        decisions[context.fold.fold_id] = decision
        selection_results[context.fold.fold_id] = evidence
        for candidate in candidates:
            if candidate.candidate_id != decision["selected_candidate"]:
                record = {
                    "schema_version": "1.0", "task_id": "24.2",
                    "evidence_id": f"{execution_id}-{context.fold.fold_id}-{candidate.safe_id}-inner-not-selected",
                    "reason": "inner_oof_underperformance", "candidate_id": candidate.candidate_id,
                    "outer_fold_id": context.fold.fold_id, "held_out_variety": context.fold.validation_variety,
                    "tested_configuration": {**asdict(candidate), "config_hash": candidate.config_hash},
                    "candidate_inner_score": evidence[candidate.candidate_id]["score"],
                    "selected_candidate": decision["selected_candidate"],
                    "selected_inner_score": decision["candidate_scores"][decision["selected_candidate"]],
                    "outer_metrics_used": False, "created_utc": _utc_now(),
                }
                _append_once(execution_root / "negative-evidence.jsonl", record)
                _append_once(TASK_ROOT / "negative-evidence.jsonl", record)

    # All four immutable decisions exist before any outer prediction is generated.
    if len(decisions) != 4 or any(not (execution_root / "decisions" / f"{context.fold.fold_id}.json").exists() for context in contexts):
        raise RuntimeError("four frozen outer decisions are required before outer evaluation")
    outer_results: list[Mapping[str, Any]] = []
    failures: list[Mapping[str, Any]] = []
    for context in contexts:
        decision = decisions[context.fold.fold_id]
        for candidate in candidates:
            selected_epoch = int(decision["candidate_selected_epochs"][candidate.candidate_id])
            for seed in SEEDS:
                try:
                    outer_results.append(_run_outer_job(
                        execution_id=execution_id, execution_root=execution_root, context=context,
                        candidate=candidate, selected_epoch=selected_epoch, decision=decision, seed=seed,
                        data=data, protocol=protocol, template=template, planning=planning,
                        environment=environment, device=device,
                    ))
                except Exception as exc:
                    failure = {"candidate_id": candidate.candidate_id, "fold_id": context.fold.fold_id,
                               "seed": seed, "error": repr(exc), "created_utc": _utc_now()}
                    failures.append(failure)
                    record = {"schema_version": "1.0", "task_id": "24.2",
                              "evidence_id": f"{execution_id}-{candidate.safe_id}-{context.fold.fold_id}-seed-{seed}-failed",
                              "reason": "failed_or_unstable", "tested_configuration": asdict(candidate),
                              "details": failure, "created_utc": _utc_now()}
                    _append_once(execution_root / "negative-evidence.jsonl", record)
                    _append_once(TASK_ROOT / "negative-evidence.jsonl", record)
                    print(json.dumps({"stage": "task_24_2_outer_failed", **failure}), file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"task 24.2 has {len(failures)} failed outer jobs; resume verified boundaries")

    candidate_summaries, policy_summary = _aggregate_results(
        execution_root=execution_root, sample_ids=sample_ids, contexts=contexts,
        candidates=candidates, decisions=decisions, results=outer_results,
    )
    for candidate in candidates:
        summary = candidate_summaries[candidate.candidate_id]
        if summary["aggregate_metrics"]["stability"]["unstable"]:
            record = {"schema_version": "1.0", "task_id": "24.2",
                      "evidence_id": f"{execution_id}-{candidate.safe_id}-seed-instability",
                      "reason": "seed_instability", "candidate_id": candidate.candidate_id,
                      "tested_configuration": summary["configuration"],
                      "candidate_metrics": summary["aggregate_metrics"],
                      "details": {"threshold": INSTABILITY_STD_THRESHOLD}, "created_utc": _utc_now()}
            _append_once(execution_root / "negative-evidence.jsonl", record)
            _append_once(TASK_ROOT / "negative-evidence.jsonl", record)

    policy_score = float(policy_summary["aggregate_metrics"]["targets"]["Gsw"]["pooled_oof_r2_seed_mean"])
    controls_complete = all(len(candidate_summaries[candidate.candidate_id]["member_run_ids"]) == 12 for candidate in candidates)
    claim_decision = _claim(policy_score, controls_complete=controls_complete)
    promotion = {
        "evaluator": "benchmark.reporting.ClaimGate", "decision": claim_decision,
        "promotion_permitted": claim_decision["status"] == "primary_strict",
        "outer_metrics_used_to_change_selection": False,
        "legacy_task_14_12_or_14_13_metrics_used_to_change_selection": False,
        "required_candidates_complete": controls_complete,
        "official_promotion_checkpoint_created": False,
        "note": "Task 24.2 repairs nested LOVO evidence only; task 24.6 remains responsible for the official promotion checkpoint.",
    }
    _publish_json_once(execution_root / "claim-and-promotion.json", promotion)

    supersession_records = []
    for legacy in legacy_reports:
        record = {
            "schema_version": "1.0", "task_id": "24.2",
            "evidence_id": f"{execution_id}-supersedes-{hashlib.sha256(legacy['report_path'].encode('utf-8')).hexdigest()[:16]}",
            "relationship": "supersedes_for_leakage_safe_nested_lovo_claim",
            "prior_record": legacy, "prior_record_preserved": True,
            "prior_claim_status": "development", "new_report": report_path.relative_to(WORKSPACE).as_posix(),
            "new_claim_status": claim_decision["status"], "created_utc": _utc_now(),
        }
        supersession_records.append(record)

    readonly_after = {
        "src": _tree_fingerprint(WORKSPACE / "src"), "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "training_dataset_sources": _training_sources_identity(manifest), "train_manifest": hash_file(TRAIN_MANIFEST),
        "legacy_evidence": hash_object(legacy_reports), "fast_lane_templates": _tree_fingerprint(WORKSPACE / "configs" / "fast_lane"),
        "caches": _tree_fingerprint(WORKSPACE / "caches"), "checkpoints": _tree_fingerprint(WORKSPACE / "checkpoints"),
    }
    if readonly_before != readonly_after:
        raise RuntimeError("a read-only source, legacy artifact, dataset, cache, or template changed")

    report = {
        "schema_version": "1.0", "task_id": "24.2", "status": "complete",
        "execution_id": execution_id, "command": COMMAND_EXECUTE, "completed_utc": _utc_now(),
        "profile": plan["profile"],
        "identity": {
            "dataset_hash": manifest.dataset_hash, "protocol_hash": protocol.protocol_hash,
            "plan_hash": plan["plan_hash"], "candidate_set_hash": plan["candidate_set_hash"],
            "environment_lock_hash": environment.lock_hash, "runner_hash": hash_file(Path(__file__)),
            "model_source_hash": hash_file(WORKSPACE / "benchmark" / "models" / "hyperleaf_candidates.py"),
            "preprocessing_source_hash": hash_file(WORKSPACE / "benchmark" / "transforms" / "preprocessing.py"),
        },
        "selection_decisions": decisions,
        "selection_evidence": selection_results,
        "nested_policy": policy_summary,
        "candidate_control_table": candidate_summaries,
        "claim_and_promotion": promotion,
        "leakage_audit": {
            "outer_decision_count": len(decisions),
            "all_decision_sources_equal_outer_train": all(set(decision["source_sample_ids"]) == set(next(context.fold.train_ids for context in contexts if context.fold.fold_id == fold_id)) for fold_id, decision in decisions.items()),
            "all_decision_outer_overlap_counts": {fold_id: decision["outer_test_overlap_count"] for fold_id, decision in decisions.items()},
            "inner_preprocessing_fit_ids_verified": True, "outer_preprocessing_fit_ids_verified": True,
            "decisions_frozen_before_outer_evaluation": True,
            "outer_evaluation_exactly_once_per_candidate_fold_seed": len(outer_results) == len(candidates) * len(contexts) * len(SEEDS) and all(int(result["outer_successful_evaluations"]) == 1 for result in outer_results),
            "task_14_12_scores_used": False, "task_14_13_outer_metrics_used": False,
            "new_outer_metrics_fed_back": False,
        },
        "resource_use": {
            "wall_clock_duration_ms": int((time.monotonic_ns() - overall_started_ns) // 1_000_000),
            "peak_device_memory_bytes": max(int(result["runtime_measurements"]["peak_device_memory_bytes"]) for result in outer_results),
            "outer_completed_epochs_total": sum(int(result["runtime_measurements"]["completed_epochs"]) for result in outer_results),
            "outer_completed_training_steps_total": sum(int(result["runtime_measurements"]["completed_training_steps"]) for result in outer_results),
            "inner_completed_training_steps_total": sum(int(result["completed_training_steps"]) for values in selection_results.values() for result in values.values()),
            "input_volume_bytes_actual_reads": data.input_volume_bytes,
            "budget": template.resource_budget.model_dump(mode="json"),
        },
        "scope": {
            "device": "cuda:0", "device_name": torch.cuda.get_device_name(0), "process_count": 1,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "precision": "float32", "execution_mode": "deterministic", "autocast": False,
            "official_test_images_read": False, "official_test_labels_read": False,
            "solution_csv_read": False, "official_evaluation_budget_consumed": 0,
            "read_only_inputs_unchanged": True,
        },
        "legacy_evidence": legacy_reports,
        "supersession_links": supersession_records,
        "artifacts": {
            "frozen_plan": "frozen-plan.json", "protocol": "protocol.canonical.json",
            "environment_lock": "environment-lock.json", "decisions": "decisions/",
            "inner_oof": "selection/", "outer_jobs": f"{template.output_root}/jobs/",
            "recovery": template.recovery_root, "aggregates": "aggregates/",
            "negative_evidence": "negative-evidence.jsonl", "claim": "claim-and-promotion.json",
        },
        "caveats": [
            "Task 14.12 and 14.13 artifacts remain unchanged and are retained as development evidence.",
            "Outer metrics estimate performance only and were not used to change candidates, epochs, routes, losses, transforms, or budgets.",
            "The task-24.2 nested-policy may select different predeclared candidates for different held-out varieties, as required by nested evaluation.",
            "The official HyperLeaf test partition and solution.csv remained sealed; official budget consumption is zero.",
            "Task 24.6, not this remediation task, controls any later official promotion checkpoint.",
        ],
        "blockers": [] if claim_decision["status"] == "primary_strict" else list(claim_decision["blocking_reasons"]),
    }
    _publish_json_once(report_path, report)
    for record in supersession_records:
        completed = {**record, "new_report_hash": hash_file(report_path)}
        _append_once(execution_root / "supersession-links.jsonl", completed)
        _append_once(TASK_ROOT / "supersession-links.jsonl", completed)
    summary = {
        "status": "complete", "execution_id": execution_id,
        "report": report_path.relative_to(WORKSPACE).as_posix(),
        "claim_status": claim_decision["status"], "promotion_permitted": promotion["promotion_permitted"],
        "selected_candidate_by_outer": policy_summary["selected_candidate_by_outer"],
        "selected_epoch_by_outer": policy_summary["selected_epoch_by_outer"],
        "official_evaluation_budget_consumed": 0,
    }
    _atomic_json(TASK_ROOT / "latest-task-24.2-summary.json", summary)
    _publish_json_once(ARTIFACT_ROOT / execution_id / "report-reference.json", {
        "schema_version": "1.0", "task_id": "24.2", "execution_id": execution_id,
        "report": report_path.relative_to(WORKSPACE).as_posix(), "report_hash": hash_file(report_path),
        "claim_status": claim_decision["status"],
    })
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute a new governed task-24.2 nested LOVO run")
    parser.add_argument("--resume", metavar="EXECUTION_ID", help="resume one exact profile-bound task-24.2 execution")
    arguments = parser.parse_args(argv)
    if arguments.execute == bool(arguments.resume):
        parser.error("choose exactly one of --execute or --resume EXECUTION_ID")
    summary = execute_nested(resume_execution_id=arguments.resume)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
