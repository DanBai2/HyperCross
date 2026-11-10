"""Execute task 24.4 HyperLeaf nested-inner-only optimization.

The runner consumes the immutable task-24.3 registry, materializes labels and
inputs only for the current outer context's three permitted varieties, trains
the complete ten-entry paired screen, advances at most three optimization
families plus their declared guards, and never performs outer evaluation.
"""
from __future__ import annotations

import argparse
import csv
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

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS
from benchmark.catalog.raw_cube import open_tiff_cube
from benchmark.models.hyperleaf_candidates import (
    AVERAGE_BRANCH,
    INDEPENDENT_TARGET_MODEL_KEY,
    PCA_BRANCH,
    PCA_INDEPENDENT_TARGET_MODEL_KEY,
    RGB_BRANCH,
    RGB_PCA_CONSISTENCY_MODEL_KEY,
    rgb_pca_consistency_loss,
)
from benchmark.models.hyperleaf_optimization import (
    HETEROSCEDASTIC_MODEL_KEY,
    LEARNED_GATED_MODEL_KEY,
    create_hyperleaf_optimization_model_registry,
)
from benchmark.protocols.hyperleaf_lovo import plan_hyperleaf_lovo
from benchmark.protocols.planner import ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.remediation import (
    RemediationPlanningContext,
    RemediationProfile,
    load_remediation_template,
)
from benchmark.schema import (
    EnvironmentLock,
    ExecutionMode,
    FitBoundary,
    FitBoundaryScope,
    MetricDirection,
    NumericPrecision,
    RuntimeMeasurements,
    assess_resource_budget,
    capture_environment_lock,
    canonical_serialize,
    hash_file,
    hash_object,
)
from benchmark.transforms.hyperleaf_candidates import (
    HYPERLEAF_STYLE_CONFIGS,
    apply_hyperleaf_style,
)
from benchmark.transforms.hyperleaf_optimization import (
    HETEROSCEDASTIC_LOSS,
    MSE_LOSS,
    STANDARD_SPECTRAL_NORMALIZATION,
    TARGET_BALANCED_LOSS,
    VARIETY_ROBUST_SPECTRAL_NORMALIZATION,
    fit_target_balance_weights,
    fit_variety_robust_spectral_normalization,
    heteroscedastic_regression_loss,
    target_balanced_regression_loss,
)
from benchmark.transforms.preprocessing import PreprocessingManager

from .hyperleaf_nested_remediation import OuterContext, build_nested_contexts
from .hyperleaf_optimization_registry import (
    CandidateRole,
    HyperLeafOptimizationCandidate,
    HyperLeafOptimizationPlan,
    InnerOofPromotionEvidence,
    SeedTargetR2,
    build_hyperleaf_optimization_plan,
    evaluate_promotion,
)
from .hyperleaf_screening import (
    BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    TRAIN_MANIFEST,
    WEIGHT_DECAY,
    ScreeningFold,
    _atomic_bytes,
    _atomic_json,
    _atomic_torch,
    _load_train_manifest,
    compute_grouped_metrics,
)

WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
PROFILE_PATH = WORKSPACE / "configs" / "remediation" / "v1" / "deterministic-float32.json"
PREREGISTRATION_PATH = WORKSPACE / "configs" / "optimization" / "v1" / "hyperleaf-task-24.3-preregistration.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "optimization" / "hyperleaf" / "v1" / "deterministic-float32"
RECOVERY_ROOT = OUTPUT_ROOT / "recovery"
ARTIFACT_ROOT = WORKSPACE / "artifacts" / "optimization" / "hyperleaf" / "v1" / "deterministic-float32"
BASELINE_DECISIONS_ROOT = (
    WORKSPACE / "runs_v2" / "remediation" / "v1" / "deterministic-float32"
    / "hyperleaf-nested" / "executions" / "task-24-2-20261027T101633949873Z" / "decisions"
)
SEEDS = (7, 42, 123)
SCREEN_CAP = 20
PROMOTION_CAP = 80
STYLE_KEY = "mild"
COMMAND_EXECUTE = "conda run --no-capture-output -n TCFM python -m benchmark.runners.hyperleaf_inner_optimization --execute"
COMMAND_RESUME = "conda run --no-capture-output -n TCFM python -m benchmark.runners.hyperleaf_inner_optimization --resume {execution_id}"
TARGET_KEYS = ("GrainWeight", "Gsw", "PhiPS2")


@dataclass
class ContextData:
    sample_ids: tuple[str, ...]
    id_to_index: Mapping[str, int]
    varieties: tuple[str, ...]
    targets: np.ndarray
    rgb: np.ndarray
    standard_spectrum: np.ndarray
    robust_spectrum_by_fold: Mapping[str, np.ndarray]
    pca_by_fold: Mapping[str, np.ndarray]
    target_artifacts: Mapping[str, Any]
    pca_artifacts: Mapping[str, Any]
    robust_artifacts: Mapping[str, Any]
    target_balance_weights: Mapping[str, np.ndarray]
    input_volume_bytes: int
    permitted_label_ids_hash: str


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


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


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


def _safe_id(candidate_id: str) -> str:
    return hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]


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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("task 24.4 requires exactly one visible CUDA device")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5080":
        raise RuntimeError("task 24.4 requires the local RTX 5080")
    return environment


def _replace_adaptive_pool(module: nn.Module) -> int:
    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool1d) and child.output_size == 8:
            setattr(module, name, nn.AvgPool1d(kernel_size=2, stride=2))
            replaced += 1
        else:
            replaced += _replace_adaptive_pool(child)
    return replaced


def _load_permitted_targets(
    allowed_ids: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, Mapping[str, Any]]:
    """Materialize target fields only after an ID passes the outer-train capability."""

    allowed = set(allowed_ids)
    rows: dict[str, tuple[float, float, float]] = {}
    path = DATA_ROOT / "train.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        header_line = stream.readline()
        header = next(csv.reader([header_line]))
        expected = ["ImageId", *TARGET_KEYS]
        if header[:4] != expected:
            raise RuntimeError("HyperLeaf train.csv leading columns differ from frozen contract")
        for raw_line in stream:
            # ImageId is the first field. Do not parse or index any target field for
            # a held-out ID; the remainder stays an opaque string in this process.
            sample_id, separator, _opaque_remainder = raw_line.partition(",")
            if not separator or sample_id not in allowed:
                continue
            parsed = next(csv.reader([raw_line]))
            values = tuple(float(parsed[index]) for index in range(1, 4))
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite permitted target: {sample_id}")
            rows[sample_id] = values  # type: ignore[assignment]
    ordered = tuple(sorted(allowed, key=str.encode))
    if set(rows) != allowed:
        raise RuntimeError("permitted target IDs do not exactly cover the outer-train boundary")
    targets = np.asarray([rows[sample_id] for sample_id in ordered], dtype=np.float64)
    audit = {
        "label_source": "G:/Datasets/HyperLeaf2024/train.csv",
        "materialized_sample_ids": ordered,
        "materialized_sample_ids_hash": hash_object(ordered),
        "materialized_count": len(ordered),
        "held_out_target_fields_materialized": 0,
        "official_target_fields_materialized": 0,
    }
    return ordered, targets, audit


def _resize_chw(values: np.ndarray, size: tuple[int, int] = (32, 64)) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)
    return (
        F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
        .squeeze(0)
        .numpy()
        .astype(np.float32, copy=False)
    )


def _load_context_data(
    *,
    manifest: Any,
    protocol_hash: str,
    context: OuterContext,
    execution_root: Path,
) -> ContextData:
    allowed_ids = tuple(sorted(context.fold.train_ids, key=str.encode))
    if set(allowed_ids) & set(context.fold.validation_ids):
        raise RuntimeError("outer held-out IDs entered the permitted materialization boundary")
    sample_ids, targets, access_audit = _load_permitted_targets(allowed_ids)
    records = {record.sample_id: record for record in manifest.records}
    varieties = tuple(str(records[sample_id].group_id) for sample_id in sample_ids)
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    raw_spectra: list[np.ndarray] = []
    input_volume_bytes = (DATA_ROOT / "train.csv").stat().st_size
    for position, sample_id in enumerate(sample_ids):
        path = DATA_ROOT / Path(records[sample_id].input_references["full_cube"])
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError("training input left the approved G: data root")
        input_volume_bytes += path.stat().st_size
        with open_tiff_cube(path, axis_order="BHW") as reader:
            raw = np.asarray(reader.memmap_view(), dtype=np.float64).mean(axis=(1, 2))
        if not np.isfinite(raw).all():
            raise RuntimeError(f"non-finite training cube: {sample_id}")
        raw_spectra.append(raw)
        if (position + 1) % 200 == 0 or position + 1 == len(sample_ids):
            print(json.dumps({"stage": "task_24_4_context_spectra", "outer_fold_id": context.fold.fold_id,
                              "completed": position + 1, "total": len(sample_ids)}), flush=True)
    raw_spectrum_array = np.stack(raw_spectra)
    interpolated_raw = np.stack(
        [np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, row.size), row) for row in raw_spectrum_array]
    ).astype(np.float32)

    manager = PreprocessingManager()
    target_artifacts: dict[str, Any] = {}
    pca_artifacts: dict[str, Any] = {}
    robust_artifacts: dict[str, Any] = {}
    target_balance_weights: dict[str, np.ndarray] = {}
    boundaries: dict[str, FitBoundary] = {}
    for fold in context.inner_folds:
        train_indices = np.asarray([id_to_index[sample_id] for sample_id in fold.train_ids], dtype=np.int64)
        boundary = FitBoundary(
            fold_id=context.fold.fold_id,
            scope=FitBoundaryScope.INNER_TRAIN,
            inner_fold_id=fold.fold_id,
            allowed_fit_sample_ids=fold.train_ids,
            outer_test_sample_ids=context.fold.validation_ids,
        )
        target = manager.fit_target_normalization(
            targets[train_indices], fold.train_ids, boundary,
            dataset_hash=manifest.dataset_hash, protocol_hash=protocol_hash,
        )
        pca = manager.fit_pca(
            raw_spectrum_array[train_indices], fold.train_ids, boundary,
            dataset_hash=manifest.dataset_hash, protocol_hash=protocol_hash,
            n_components=3, whiten=False,
        )
        robust = fit_variety_robust_spectral_normalization(
            torch.from_numpy(interpolated_raw[train_indices]),
            sample_ids=fold.train_ids,
            variety_ids=tuple(varieties[index] for index in train_indices),
            allowed_sample_ids=fold.train_ids,
            dataset_hash=manifest.dataset_hash,
            protocol_hash=protocol_hash,
            fold_id=fold.fold_id,
        )
        weights = fit_target_balance_weights(
            torch.from_numpy(targets[train_indices].astype(np.float32))
        ).numpy()
        for fit_ids in (target.fit_sample_ids, pca.fit_sample_ids, robust.fit_sample_ids):
            if set(fit_ids) != set(fold.train_ids) or set(fit_ids) & set(context.fold.validation_ids):
                raise RuntimeError(f"fitted state crossed inner boundary: {fold.fold_id}")
        target_artifacts[fold.fold_id] = target
        pca_artifacts[fold.fold_id] = pca
        robust_artifacts[fold.fold_id] = robust
        target_balance_weights[fold.fold_id] = weights
        boundaries[fold.fold_id] = boundary
        transform_root = execution_root / "transforms" / context.fold.fold_id / fold.fold_id
        _publish_json_once(transform_root / "target-normalization.json", target.model_dump(mode="json"))
        _publish_json_once(transform_root / "pca.json", pca.model_dump(mode="json"))
        _publish_json_once(transform_root / "variety-robust-spectrum.json", asdict(robust))
        _publish_json_once(transform_root / "target-balance.json", {
            "fit_sample_ids": tuple(sorted(fold.train_ids, key=str.encode)),
            "fit_sample_ids_hash": hash_object(tuple(sorted(fold.train_ids, key=str.encode))),
            "weights": [float(value) for value in weights],
            "loss_key": TARGET_BALANCED_LOSS,
        })

    standard_spectrum_rows: list[np.ndarray] = []
    robust_spectrum_rows: dict[str, list[np.ndarray]] = {fold.fold_id: [] for fold in context.inner_folds}
    rgb_rows: list[np.ndarray] = []
    pca_rows: dict[str, list[np.ndarray]] = {fold.fold_id: [] for fold in context.inner_folds}
    for position, sample_id in enumerate(sample_ids):
        raw_interp = interpolated_raw[position]
        scale = float(raw_interp.std())
        standard_spectrum_rows.append(
            ((raw_interp - raw_interp.mean()) / (scale if scale > 1.0e-12 else 1.0)).astype(np.float32)
        )
        for fold in context.inner_folds:
            transformed = robust_artifacts[fold.fold_id].transform(
                torch.from_numpy(raw_interp[None, :])
            )[0]
            robust_spectrum_rows[fold.fold_id].append(transformed.numpy().astype(np.float32))
        path = DATA_ROOT / Path(records[sample_id].input_references["full_cube"])
        input_volume_bytes += path.stat().st_size
        with open_tiff_cube(path, axis_order="BHW") as reader:
            cube = reader.memmap_view()
            bands = reader.shape[0]
            band_indices = tuple(min(bands - 1, int(bands * ratio)) for ratio in (0.72, 0.48, 0.20))
            rgb = np.stack([np.asarray(cube[index], dtype=np.float32) for index in band_indices])
            rgb = rgb / (65535.0 if float(np.nanmax(rgb)) > 1.0 else 1.0)
            image = Image.fromarray(np.moveaxis((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 0, -1))
            resized = image.resize((64, 32), Image.Resampling.BILINEAR)
            rgb_rows.append(np.moveaxis(np.asarray(resized, dtype=np.float32) / 255.0, -1, 0))
            pixels = np.moveaxis(np.asarray(cube, dtype=np.float32), 0, -1).reshape(-1, bands)
            for fold in context.inner_folds:
                projected = manager.apply(pca_artifacts[fold.fold_id], pixels, boundaries[fold.fold_id])
                projected = projected.reshape(reader.shape[1], reader.shape[2], 3)
                minimum = projected.min((0, 1), keepdims=True)
                maximum = projected.max((0, 1), keepdims=True)
                denominator = np.where(maximum - minimum > 1.0e-12, maximum - minimum, 1.0)
                pca_rows[fold.fold_id].append(
                    _resize_chw(np.moveaxis(((projected - minimum) / denominator).astype(np.float32), -1, 0))
                )
        if (position + 1) % 100 == 0 or position + 1 == len(sample_ids):
            print(json.dumps({"stage": "task_24_4_context_representations", "outer_fold_id": context.fold.fold_id,
                              "completed": position + 1, "total": len(sample_ids)}), flush=True)

    data = ContextData(
        sample_ids=sample_ids,
        id_to_index=id_to_index,
        varieties=varieties,
        targets=targets,
        rgb=np.stack(rgb_rows).astype(np.float32),
        standard_spectrum=np.stack(standard_spectrum_rows).astype(np.float32),
        robust_spectrum_by_fold={key: np.stack(value).astype(np.float32) for key, value in robust_spectrum_rows.items()},
        pca_by_fold={key: np.stack(value).astype(np.float32) for key, value in pca_rows.items()},
        target_artifacts=target_artifacts,
        pca_artifacts=pca_artifacts,
        robust_artifacts=robust_artifacts,
        target_balance_weights=target_balance_weights,
        input_volume_bytes=input_volume_bytes,
        permitted_label_ids_hash=str(access_audit["materialized_sample_ids_hash"]),
    )
    _publish_json_once(execution_root / "label-access" / f"{context.fold.fold_id}.json", {
        **access_audit,
        "outer_fold_id": context.fold.fold_id,
        "held_out_variety": context.fold.validation_variety,
        "outer_test_overlap_count": len(set(sample_ids) & set(context.fold.validation_ids)),
        "input_sample_count": len(sample_ids),
        "input_volume_bytes": input_volume_bytes,
    })
    return data


def _build_state(candidate: HyperLeafOptimizationCandidate, fold: ScreeningFold, seed: int, device: torch.device) -> TrainState:
    _set_determinism(seed)
    model = create_hyperleaf_optimization_model_registry().build(candidate.model_key)
    _replace_adaptive_pool(model)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROMOTION_CAP)
    return TrainState(fold=fold, seed=seed, model=model, optimizer=optimizer, scheduler=scheduler)


def _spectrum(candidate: HyperLeafOptimizationCandidate, data: ContextData, fold_id: str, indices: np.ndarray) -> torch.Tensor:
    source = (
        data.robust_spectrum_by_fold[fold_id]
        if candidate.spectral_normalization_key == VARIETY_ROBUST_SPECTRAL_NORMALIZATION
        else data.standard_spectrum
    )
    return torch.from_numpy(source[indices])


def _weighted_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    candidate: HyperLeafOptimizationCandidate,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    if candidate.loss_key == TARGET_BALANCED_LOSS:
        return target_balanced_regression_loss(prediction, targets, target_weights=target_weights)
    return F.mse_loss(prediction, targets)


def _forward_and_loss(
    state: TrainState,
    candidate: HyperLeafOptimizationCandidate,
    rgb: torch.Tensor,
    pca: torch.Tensor,
    spectrum: torch.Tensor,
    targets: torch.Tensor,
    target_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = state.model
    if candidate.model_key == INDEPENDENT_TARGET_MODEL_KEY:
        prediction = model(rgb, spectrum)
        return prediction, _weighted_loss(prediction, targets, candidate, target_weights)
    if candidate.model_key == PCA_INDEPENDENT_TARGET_MODEL_KEY:
        prediction = model(pca, spectrum)
        return prediction, _weighted_loss(prediction, targets, candidate, target_weights)
    if candidate.model_key == HETEROSCEDASTIC_MODEL_KEY:
        branches = model.mean_model.branch_predictions(rgb, pca, spectrum)
        prediction = torch.cat(
            tuple(branches[route][:, index:index + 1] for index, route in enumerate(candidate.target_routes)), dim=1
        )
        log_variance = model.log_variance_head(torch.cat((branches[RGB_BRANCH], branches[PCA_BRANCH]), dim=1)).clamp(-8.0, 8.0)
        loss = heteroscedastic_regression_loss(prediction, log_variance, targets)
        loss = loss + rgb_pca_consistency_loss(
            branches[RGB_BRANCH], branches[PCA_BRANCH], weight=candidate.consistency_weight
        )
        return prediction, loss
    if candidate.model_key == LEARNED_GATED_MODEL_KEY:
        prediction, branches = model(rgb, pca, spectrum, target_routes=candidate.target_routes, return_branches=True)
        branch_loss = (
            F.mse_loss(branches[RGB_BRANCH], targets) + F.mse_loss(branches[PCA_BRANCH], targets)
        ) / 2.0
        gated_gsw_loss = F.mse_loss(prediction[:, 1], targets[:, 1]) / 3.0
        penalty = rgb_pca_consistency_loss(
            branches[RGB_BRANCH], branches[PCA_BRANCH], weight=candidate.consistency_weight
        )
        return prediction, branch_loss + gated_gsw_loss + penalty
    if candidate.model_key == RGB_PCA_CONSISTENCY_MODEL_KEY:
        prediction, branches = model(
            rgb, pca, spectrum, inference_branch="target-routed",
            target_routes=candidate.target_routes, return_branches=True,
        )
        branch_loss = (
            _weighted_loss(branches[RGB_BRANCH], targets, candidate, target_weights)
            + _weighted_loss(branches[PCA_BRANCH], targets, candidate, target_weights)
        ) / 2.0
        penalty = rgb_pca_consistency_loss(
            branches[RGB_BRANCH], branches[PCA_BRANCH], weight=candidate.consistency_weight
        )
        return prediction, branch_loss + penalty
    raise RuntimeError(f"unsupported preregistered model: {candidate.model_key}")


def _predict(
    state: TrainState,
    candidate: HyperLeafOptimizationCandidate,
    data: ContextData,
    device: torch.device,
) -> np.ndarray:
    state.model.eval()
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in state.fold.validation_ids], dtype=np.int64)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), EVALUATION_BATCH_SIZE):
            selected = indices[start:start + EVALUATION_BATCH_SIZE]
            rgb = torch.from_numpy(data.rgb[selected]).to(device=device, dtype=torch.float32)
            pca = torch.from_numpy(data.pca_by_fold[state.fold.fold_id][selected]).to(device=device, dtype=torch.float32)
            spectrum = _spectrum(candidate, data, state.fold.fold_id, selected).to(device=device, dtype=torch.float32)
            dummy_targets = torch.zeros((len(selected), 3), device=device)
            weights = torch.from_numpy(data.target_balance_weights[state.fold.fold_id]).to(device=device)
            prediction, _ = _forward_and_loss(state, candidate, rgb, pca, spectrum, dummy_targets, weights)
            outputs.append(prediction.detach().cpu().numpy().astype(np.float64))
    normalized = np.concatenate(outputs)
    artifact = data.target_artifacts[state.fold.fold_id]
    mean = np.asarray(artifact.state["mean"], dtype=np.float64)
    scale = np.asarray(artifact.state["scale"], dtype=np.float64)
    result = normalized * scale + mean
    if not np.isfinite(result).all():
        raise RuntimeError(f"non-finite prediction: {candidate.candidate_id}/{state.fold.fold_id}")
    return result


def _train_epoch(
    state: TrainState,
    candidate: HyperLeafOptimizationCandidate,
    data: ContextData,
    epoch: int,
    device: torch.device,
) -> float:
    state.model.train()
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in state.fold.train_ids], dtype=np.int64)
    order = torch.randperm(
        len(indices), generator=torch.Generator().manual_seed(state.seed * 100_000 + epoch)
    ).numpy()
    style = HYPERLEAF_STYLE_CONFIGS[STYLE_KEY]
    mean = np.asarray(data.target_artifacts[state.fold.fold_id].state["mean"], dtype=np.float32)
    scale = np.asarray(data.target_artifacts[state.fold.fold_id].state["scale"], dtype=np.float32)
    target_weights = torch.from_numpy(data.target_balance_weights[state.fold.fold_id]).to(device=device)
    losses: list[float] = []
    for batch_number, start in enumerate(range(0, len(order), BATCH_SIZE)):
        selected = indices[order[start:start + BATCH_SIZE]]
        rgb = torch.from_numpy(data.rgb[selected]).to(device=device, dtype=torch.float32)
        pca = torch.from_numpy(data.pca_by_fold[state.fold.fold_id][selected]).to(device=device, dtype=torch.float32)
        spectrum = _spectrum(candidate, data, state.fold.fold_id, selected).to(device=device, dtype=torch.float32)
        targets = torch.from_numpy(((data.targets[selected] - mean) / scale).astype(np.float32)).to(device)
        augmentation_seed = state.seed * 10_000_000 + epoch * 10_000 + batch_number
        rgb, styled_spectrum = apply_hyperleaf_style(rgb, spectrum, config=style, seed=augmentation_seed)
        pca, _ = apply_hyperleaf_style(pca, spectrum, config=style, seed=augmentation_seed)
        state.optimizer.zero_grad(set_to_none=True)
        _prediction, loss = _forward_and_loss(
            state, candidate, rgb, pca, styled_spectrum, targets, target_weights
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite task-24.4 loss")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in state.model.parameters()
        ):
            raise RuntimeError("non-finite task-24.4 gradient")
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), GRADIENT_CLIP)
        state.optimizer.step()
        state.completed_steps += 1
        losses.append(float(loss.detach().item()))
    state.scheduler.step()
    return float(np.mean(losses))


def _state_identity(
    *, plan: HyperLeafOptimizationPlan, execution_id: str, context: OuterContext,
    candidate: HyperLeafOptimizationCandidate, fold: ScreeningFold, seed: int,
    environment: EnvironmentLock, data: ContextData,
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0", "task_id": "24.4", "execution_id": execution_id,
        "dataset_hash": plan.context.dataset_hash, "protocol_hash": plan.context.protocol_hash,
        "model_key": candidate.model_key, "candidate_id": candidate.candidate_id,
        "candidate_configuration_hash": candidate.configuration_hash,
        "comparison_context_hash": plan.context.comparison_context_hash,
        "outer_fold_id": context.fold.fold_id, "inner_fold_id": fold.fold_id, "seed": seed,
        "environment_lock_hash": environment.lock_hash,
        "precision_profile_config_hash": plan.context.precision_profile_config_hash,
        "precision_identity_hash": plan.context.precision_identity_hash,
        "numeric_precision": "float32", "execution_mode": "deterministic",
        "max_epoch": PROMOTION_CAP, "scheduler_t_max": PROMOTION_CAP,
        "resource_budget_hash": plan.context.resource_budget_hash,
        "metric_configuration_hash": plan.context.metric_configuration_hash,
        "input_volume_bytes": data.input_volume_bytes,
        "permitted_label_ids_hash": data.permitted_label_ids_hash,
        "official_budget": 0,
    }
    payload["frozen_identity_hash"] = hash_object(payload)
    return payload


def _checkpoint_payload(
    state: TrainState, candidate: HyperLeafOptimizationCandidate, epoch: int,
    identity_hash: str, stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "task_id": "24.4", "stage": stage,
        "candidate_id": candidate.candidate_id,
        "candidate_configuration_hash": candidate.configuration_hash,
        "fold_id": state.fold.fold_id, "seed": state.seed,
        "completed_epoch": epoch, "completed_training_steps": state.completed_steps,
        "frozen_identity_hash": identity_hash,
        "model": state.model.state_dict(), "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "stage_state": {"next_epoch": epoch + 1, "phase": stage},
    }


def _job_paths(
    execution_id: str, context: OuterContext, candidate: HyperLeafOptimizationCandidate,
    fold: ScreeningFold, seed: int,
) -> tuple[Path, Path]:
    relative = Path("executions") / execution_id / "jobs" / context.fold.fold_id / _safe_id(candidate.candidate_id) / f"seed-{seed}" / fold.fold_id
    return OUTPUT_ROOT / relative, RECOVERY_ROOT / execution_id / context.fold.fold_id / _safe_id(candidate.candidate_id) / f"seed-{seed}" / fold.fold_id


def _load_recovery_exact(
    state: TrainState, candidate: HyperLeafOptimizationCandidate, recovery_root: Path,
    epoch: int, identity_hash: str, device: torch.device,
) -> None:
    if epoch == 0:
        return
    path = recovery_root / f"slot-{epoch % 2}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    observed = (
        checkpoint["candidate_id"], checkpoint["candidate_configuration_hash"],
        checkpoint["fold_id"], checkpoint["seed"], checkpoint["completed_epoch"],
        checkpoint["frozen_identity_hash"],
    )
    expected = (candidate.candidate_id, candidate.configuration_hash, state.fold.fold_id, state.seed, epoch, identity_hash)
    if observed != expected:
        raise RuntimeError("recovery checkpoint identity or exact boundary mismatch")
    state.model.load_state_dict(checkpoint["model"], strict=True)
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.scheduler.load_state_dict(checkpoint["scheduler"])
    state.completed_steps = int(checkpoint["completed_training_steps"])


def _save_recovery(
    state: TrainState, candidate: HyperLeafOptimizationCandidate, recovery_root: Path,
    epoch: int, identity_hash: str, stage: str,
) -> None:
    payload = _checkpoint_payload(state, candidate, epoch, identity_hash, stage)
    _atomic_torch(recovery_root / f"slot-{epoch % 2}.pt", payload)
    _atomic_json(recovery_root / "latest.json", {
        "schema_version": "1.0", "frozen_identity_hash": identity_hash,
        "completed_epoch": epoch, "completed_training_steps": state.completed_steps,
        "state_names": ["model", "optimizer", "scheduler", "stage"],
    })


def _inner_predictions(
    states: Sequence[TrainState], candidate: HyperLeafOptimizationCandidate,
    data: ContextData, device: torch.device,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    by_id: dict[str, tuple[str, np.ndarray]] = {}
    for state in states:
        prediction = _predict(state, candidate, data, device)
        for sample_id, row in zip(state.fold.validation_ids, prediction):
            if sample_id in by_id:
                raise RuntimeError("inner OOF sample predicted more than once")
            by_id[sample_id] = (state.fold.fold_id, row)
    expected = {sample_id for state in states for sample_id in state.fold.validation_ids}
    if set(by_id) != expected:
        raise RuntimeError("inner OOF coverage is incomplete")
    ids = tuple(sorted(expected, key=str.encode))
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in ids], dtype=np.int64)
    return (
        ids, tuple(by_id[sample_id][0] for sample_id in ids), data.targets[indices].copy(),
        np.stack([by_id[sample_id][1] for sample_id in ids]),
    )


def _seed_metrics(
    states: Sequence[TrainState], candidate: HyperLeafOptimizationCandidate,
    data: ContextData, device: torch.device,
) -> tuple[Mapping[str, Any], tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    ids, fold_ids, truth, forecast = _inner_predictions(states, candidate, data, device)
    return compute_grouped_metrics(ids, fold_ids, truth, forecast), ids, fold_ids, truth, forecast


def _persist_oof(
    *, execution_root: Path, context: OuterContext, candidate: HyperLeafOptimizationCandidate,
    seed: int, selected_epoch: int, stage: str, states: Sequence[TrainState],
    data: ContextData, device: torch.device,
) -> tuple[str, Mapping[str, Any], str]:
    metrics, ids, fold_ids, truth, forecast = _seed_metrics(states, candidate, data, device)
    if set(ids) != set(context.fold.train_ids) or set(ids) & set(context.fold.validation_ids):
        raise RuntimeError("inner OOF artifact crossed the outer-train boundary")
    payload = {
        "schema_version": "1.0", "task_id": "24.4", "kind": "exact-inner-oof",
        "evidence_scope": "nested_inner_oof_only", "stage": stage,
        "outer_fold_id": context.fold.fold_id, "held_out_variety": context.fold.validation_variety,
        "candidate_id": candidate.candidate_id,
        "candidate_configuration_hash": candidate.configuration_hash,
        "seed": seed, "selected_epoch": selected_epoch,
        "source_sample_ids": ids, "source_sample_ids_hash": hash_object(ids),
        "held_out_overlap_count": 0, "outer_metrics_used": False,
        "official_labels_used": False,
        "records": [
            {
                "key": {"sample_id": sample_id, "fold_id": fold_id, "seed": seed,
                        "inference_branch": "target-routed"},
                "target_name": "three-target-output",
                "y_true": [float(value) for value in actual],
                "y_pred": [float(value) for value in predicted], "probability": None,
            }
            for sample_id, fold_id, actual, predicted in zip(ids, fold_ids, truth, forecast)
        ],
        "metrics": metrics,
    }
    payload["source_prediction_hash"] = hash_object(payload)
    relative = Path("selection") / context.fold.fold_id / _safe_id(candidate.candidate_id) / f"seed-{seed}-{stage}-inner-oof.json"
    path = execution_root / relative
    _publish_json_once(path, payload)
    return str(payload["source_prediction_hash"]), metrics, path.relative_to(WORKSPACE).as_posix()


def _train_to_cap(
    *, execution_id: str, execution_root: Path, plan: HyperLeafOptimizationPlan,
    context: OuterContext, candidate: HyperLeafOptimizationCandidate, data: ContextData,
    cap: int, stage: str, template: Any, environment: EnvironmentLock, device: torch.device,
) -> dict[str, Any]:
    candidate_root = execution_root / "selection" / context.fold.fold_id / _safe_id(candidate.candidate_id)
    result_path = candidate_root / f"{stage}-result.json"
    if result_path.exists():
        return _read_json(result_path)
    common_path = candidate_root / "common-recovery.json"
    states: list[TrainState] = []
    metadata: dict[tuple[int, str], tuple[Path, Path, str]] = {}
    for seed in SEEDS:
        for fold in context.inner_folds:
            state = _build_state(candidate, fold, seed, device)
            job_root, recovery_root = _job_paths(execution_id, context, candidate, fold, seed)
            identity = _state_identity(
                plan=plan, execution_id=execution_id, context=context, candidate=candidate,
                fold=fold, seed=seed, environment=environment, data=data,
            )
            identity_path = job_root / "frozen-identity.json"
            _publish_json_once(identity_path, identity)
            states.append(state)
            metadata[(seed, fold.fold_id)] = (job_root, recovery_root, str(identity["frozen_identity_hash"]))

    completed_epoch = 0
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    selected_epoch = 0
    accumulated_wall_ms = 0
    if common_path.exists():
        common = _read_json(common_path)
        if common.get("candidate_configuration_hash") != candidate.configuration_hash:
            raise RuntimeError("common recovery candidate identity mismatch")
        completed_epoch = int(common["completed_epoch"])
        if completed_epoch > cap:
            raise RuntimeError("common recovery is beyond requested immutable stage cap")
        history = list(common.get("history", []))
        best_score = float(common.get("best_score", -math.inf))
        selected_epoch = int(common.get("selected_epoch", 0))
        accumulated_wall_ms = int(common.get("accumulated_wall_clock_duration_ms", 0))
        for state in states:
            _job_root, recovery_root, identity_hash = metadata[(state.seed, state.fold.fold_id)]
            _load_recovery_exact(state, candidate, recovery_root, completed_epoch, identity_hash, device)
    if stage == "promotion-screen-80-epochs" and completed_epoch < SCREEN_CAP:
        raise RuntimeError("promotion stage cannot start before complete paired screen")

    invocation_started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    for epoch in range(completed_epoch + 1, cap + 1):
        losses: dict[str, float] = {}
        for state in states:
            losses[f"seed-{state.seed}/{state.fold.fold_id}"] = _train_epoch(
                state, candidate, data, epoch, device
            )
            _job_root, recovery_root, identity_hash = metadata[(state.seed, state.fold.fold_id)]
            _save_recovery(state, candidate, recovery_root, epoch, identity_hash, stage)
        seed_scores: dict[str, float] = {}
        seed_metrics: dict[str, Any] = {}
        for seed in SEEDS:
            metrics, _ids, _fold_ids, _truth, _forecast = _seed_metrics(
                [state for state in states if state.seed == seed], candidate, data, device
            )
            value = metrics["pooled_oof"]["targets"]["Gsw"]["r2"]
            score = -math.inf if value is None else float(value)
            seed_scores[str(seed)] = score
            seed_metrics[str(seed)] = {
                target: metrics["pooled_oof"]["targets"][target] for target in TARGET_KEYS
            }
        score = float(np.mean([seed_scores[str(seed)] for seed in SEEDS]))
        history.append({"epoch": epoch, "score": score, "seed_scores": seed_scores,
                        "seed_metrics": seed_metrics, "losses": losses})
        if score > best_score:
            best_score = score
            selected_epoch = epoch
            selected_name = "selected-screen-20.pt" if cap == SCREEN_CAP else "selected-promotion-80.pt"
            for state in states:
                job_root, _recovery_root, identity_hash = metadata[(state.seed, state.fold.fold_id)]
                _atomic_torch(job_root / "checkpoints" / selected_name,
                              _checkpoint_payload(state, candidate, epoch, identity_hash, stage))
        elapsed_ms = int((time.monotonic_ns() - invocation_started_ns) // 1_000_000)
        cumulative_ms = accumulated_wall_ms + elapsed_ms
        _atomic_json(common_path, {
            "schema_version": "1.0", "candidate_id": candidate.candidate_id,
            "candidate_configuration_hash": candidate.configuration_hash,
            "completed_epoch": epoch, "best_score": best_score, "selected_epoch": selected_epoch,
            "history": history, "accumulated_wall_clock_duration_ms": cumulative_ms,
        })
        measurements = RuntimeMeasurements(
            completed_epochs=epoch,
            completed_training_steps=max(state.completed_steps for state in states),
            wall_clock_duration_ms=cumulative_ms,
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
            device_identity=environment.device_identity,
            numeric_precision="float32",
            parameter_count=sum(parameter.numel() for parameter in states[0].model.parameters()),
            input_volume_bytes=data.input_volume_bytes,
        )
        assessment = assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch)
        if assessment.hard_exceeded:
            raise RuntimeError(f"hard resource limit exceeded at epoch boundary {epoch}")
        print(json.dumps({
            "stage": f"task_24_4_{stage}", "outer_fold_id": context.fold.fold_id,
            "candidate_id": candidate.candidate_id, "epoch": epoch,
            "target_epoch": cap, "mean_seed_gsw_r2": score,
        }), flush=True)

    if selected_epoch <= 0:
        raise RuntimeError("candidate produced no finite selected epoch")
    final_selected_name = "selected-screen-20.pt" if cap == SCREEN_CAP else "selected-promotion-80.pt"
    if cap == PROMOTION_CAP:
        missing_final = any(
            not (metadata[(state.seed, state.fold.fold_id)][0] / "checkpoints" / final_selected_name).exists()
            for state in states
        )
        if missing_final:
            # The best epoch remained in the paired screen. Preserve that exact
            # checkpoint under the final-stage retention name without retraining.
            for state in states:
                job_root, _recovery_root, _identity_hash = metadata[(state.seed, state.fold.fold_id)]
                source = job_root / "checkpoints" / "selected-screen-20.pt"
                _atomic_bytes(job_root / "checkpoints" / final_selected_name, source.read_bytes())
    for state in states:
        job_root, recovery_root, identity_hash = metadata[(state.seed, state.fold.fold_id)]
        selected = torch.load(job_root / "checkpoints" / final_selected_name, map_location=device, weights_only=False)
        if int(selected["completed_epoch"]) != selected_epoch or selected["frozen_identity_hash"] != identity_hash:
            raise RuntimeError("selected checkpoint identity mismatch")
        state.model.load_state_dict(selected["model"], strict=True)

    prediction_hashes: dict[str, str] = {}
    prediction_paths: dict[str, str] = {}
    metrics_by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        source_hash, metrics, prediction_path = _persist_oof(
            execution_root=execution_root, context=context, candidate=candidate, seed=seed,
            selected_epoch=selected_epoch, stage=stage,
            states=[state for state in states if state.seed == seed], data=data, device=device,
        )
        prediction_hashes[str(seed)] = source_hash
        prediction_paths[str(seed)] = prediction_path
        metrics_by_seed[str(seed)] = metrics

    checkpoint_refs: dict[str, Any] = {}
    last_name = "last-screen-20.pt" if cap == SCREEN_CAP else "last-promotion-80.pt"
    for state in states:
        job_root, recovery_root, _identity_hash = metadata[(state.seed, state.fold.fold_id)]
        recovery_path = recovery_root / f"slot-{cap % 2}.pt"
        _atomic_bytes(job_root / "checkpoints" / last_name, recovery_path.read_bytes())
        checkpoint_refs[f"seed-{state.seed}/{state.fold.fold_id}"] = {
            "selected": {"path": (job_root / "checkpoints" / final_selected_name).relative_to(WORKSPACE).as_posix(),
                         "hash": hash_file(job_root / "checkpoints" / final_selected_name)},
            "last": {"path": (job_root / "checkpoints" / last_name).relative_to(WORKSPACE).as_posix(),
                     "hash": hash_file(job_root / "checkpoints" / last_name)},
            "recovery": {"path": recovery_path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(recovery_path)},
            "identity": {"path": (job_root / "frozen-identity.json").relative_to(WORKSPACE).as_posix(),
                         "hash": hash_file(job_root / "frozen-identity.json")},
        }
    common = _read_json(common_path)
    result = {
        "schema_version": "1.0", "task_id": "24.4", "status": "complete",
        "stage": stage, "epoch_cap": cap, "outer_fold_id": context.fold.fold_id,
        "held_out_variety": context.fold.validation_variety,
        "candidate_id": candidate.candidate_id,
        "candidate_configuration_hash": candidate.configuration_hash,
        "selected_epoch": selected_epoch, "score": best_score,
        "source_sample_ids": tuple(sorted(context.fold.train_ids, key=str.encode)),
        "source_sample_ids_hash": hash_object(tuple(sorted(context.fold.train_ids, key=str.encode))),
        "held_out_overlap_count": 0, "outer_metrics_used": False, "official_labels_used": False,
        "source_prediction_hashes": prediction_hashes,
        "prediction_artifacts": prediction_paths,
        "per_seed_inner_metrics": metrics_by_seed,
        "history_hash": hash_object(history),
        "completed_training_steps": sum(state.completed_steps for state in states),
        "checkpoints": checkpoint_refs,
        "runtime_measurements": {
            "completed_epochs_per_fold_seed": cap,
            "completed_training_steps_total": sum(state.completed_steps for state in states),
            "wall_clock_duration_ms": int(common["accumulated_wall_clock_duration_ms"]),
            "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "device_identity": environment.device_identity,
            "numeric_precision": "float32",
            "parameter_count": sum(parameter.numel() for parameter in states[0].model.parameters()),
            "input_volume_bytes": data.input_volume_bytes,
        },
    }
    result["result_hash"] = hash_object(result)
    _publish_json_once(result_path, result)
    return result


def _promotion_evidence(
    plan: HyperLeafOptimizationPlan, candidate_id: str, result: Mapping[str, Any]
) -> InnerOofPromotionEvidence:
    rows = []
    hashes = []
    for seed in SEEDS:
        targets = result["per_seed_inner_metrics"][str(seed)]["pooled_oof"]["targets"]
        rows.append(SeedTargetR2(
            seed=seed,
            grainweight=float(targets["GrainWeight"]["r2"]),
            gsw=float(targets["Gsw"]["r2"]),
            phips2=float(targets["PhiPS2"]["r2"]),
        ))
        hashes.append(str(result["source_prediction_hashes"][str(seed)]))
    return InnerOofPromotionEvidence.create(
        candidate_id=candidate_id,
        comparison_context_hash=plan.context.comparison_context_hash,
        evidence_scope="nested_inner_oof_only",
        seed_metrics=tuple(rows), source_prediction_hashes=tuple(hashes),
        outer_metrics_used=False, official_labels_used=False,
    )


def _deltas(
    candidate: InnerOofPromotionEvidence, guard: InnerOofPromotionEvidence
) -> dict[str, Any]:
    candidate_rows = {row.seed: row for row in candidate.seed_metrics}
    guard_rows = {row.seed: row for row in guard.seed_metrics}
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {
            target: float(getattr(candidate_rows[seed], attribute) - getattr(guard_rows[seed], attribute))
            for target, attribute in (("GrainWeight", "grainweight"), ("Gsw", "gsw"), ("PhiPS2", "phips2"))
        }
    means = {target: float(np.mean([per_seed[str(seed)][target] for seed in SEEDS])) for target in TARGET_KEYS}
    return {"per_seed_delta_r2": per_seed, "mean_delta_r2": means}


def _screen_decision(
    *, plan: HyperLeafOptimizationPlan, context: OuterContext,
    screen_results: Mapping[str, Mapping[str, Any]], execution_id: str,
) -> dict[str, Any]:
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    comparisons: dict[str, Any] = {}
    eligible: list[tuple[float, str]] = []
    for candidate in plan.candidates:
        if candidate.role is not CandidateRole.OPTIMIZATION:
            continue
        candidate_evidence = _promotion_evidence(plan, candidate.candidate_id, screen_results[candidate.candidate_id])
        guard_evidence = _promotion_evidence(plan, candidate.matched_guard_id or "", screen_results[candidate.matched_guard_id or ""])
        delta = _deltas(candidate_evidence, guard_evidence)
        provisional_pass = (
            delta["mean_delta_r2"]["Gsw"] > 0.0
            and all(delta["per_seed_delta_r2"][str(seed)]["Gsw"] > 0.0 for seed in SEEDS)
            and delta["mean_delta_r2"]["GrainWeight"] >= -0.0200
            and delta["mean_delta_r2"]["PhiPS2"] >= -0.0200
        )
        comparisons[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id, "matched_guard_id": candidate.matched_guard_id,
            **delta, "provisional_guard_passed": provisional_pass,
            "candidate_evidence_hash": candidate_evidence.evidence_hash,
            "guard_evidence_hash": guard_evidence.evidence_hash,
        }
        if provisional_pass:
            eligible.append((float(delta["mean_delta_r2"]["Gsw"]), candidate.candidate_id))
    ranked = sorted(eligible, key=lambda item: (-item[0], item[1].encode("utf-8")))
    advanced_candidates = tuple(candidate_id for _score, candidate_id in ranked[:3])
    advanced_all = set(advanced_candidates)
    frontier = list(advanced_candidates)
    while frontier:
        candidate_id = frontier.pop()
        guard_id = by_id[candidate_id].matched_guard_id
        if guard_id is not None and guard_id not in advanced_all:
            advanced_all.add(guard_id)
            frontier.append(guard_id)
    decision = {
        "schema_version": "1.0", "task_id": "24.4",
        "decision_id": f"{execution_id}-{context.fold.fold_id}-paired-screen",
        "outer_fold_id": context.fold.fold_id,
        "source_sample_ids": tuple(sorted(context.fold.train_ids, key=str.encode)),
        "source_sample_ids_hash": hash_object(tuple(sorted(context.fold.train_ids, key=str.encode))),
        "held_out_overlap_count": 0,
        "complete_candidate_set": tuple(candidate.candidate_id for candidate in plan.candidates),
        "comparisons": comparisons, "ranking": [candidate_id for _score, candidate_id in ranked],
        "advanced_optimization_candidates": advanced_candidates,
        "advanced_candidates_and_guards": tuple(sorted(advanced_all, key=str.encode)),
        "maximum_new_candidate_families_advanced": 3,
        "outer_metrics_available_at_freeze": False,
        "task_14_13_metrics_used": False, "task_24_2_outer_metrics_used": False,
        "official_labels_used": False, "solution_csv_used": False,
        "tie_break_rule": plan.context.tie_break_rule, "frozen_utc": _utc_now(),
    }
    decision["decision_hash"] = hash_object(decision)
    return decision


def _fallback_epoch(context: OuterContext) -> tuple[int, str]:
    path = BASELINE_DECISIONS_ROOT / f"{context.fold.fold_id}.json"
    payload = _read_json(path)
    if payload.get("outer_metrics_available_at_freeze") is not False:
        raise RuntimeError("task-24.2 fallback decision was not frozen before outer metrics")
    candidate_id = "branch::rgb-pca-consistency"
    epoch = int(payload["candidate_selected_epochs"][candidate_id])
    return epoch, hash_file(path)


def _final_decision(
    *, plan: HyperLeafOptimizationPlan, context: OuterContext,
    screen_decision: Mapping[str, Any], final_results: Mapping[str, Mapping[str, Any]],
    execution_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    guard_decisions: dict[str, Any] = {}
    negative: list[dict[str, Any]] = []
    passed: list[tuple[float, str]] = []
    for candidate_id in screen_decision["advanced_optimization_candidates"]:
        candidate = by_id[candidate_id]
        candidate_evidence = _promotion_evidence(plan, candidate_id, final_results[candidate_id])
        guard_evidence = _promotion_evidence(plan, candidate.matched_guard_id or "", final_results[candidate.matched_guard_id or ""])
        decision = evaluate_promotion(plan, candidate_evidence, guard_evidence)
        delta = _deltas(candidate_evidence, guard_evidence)
        record = {**decision.model_dump(mode="json"), **delta}
        guard_decisions[candidate_id] = record
        if decision.passed:
            score = float(np.mean([
                row.gsw for row in candidate_evidence.seed_metrics
            ]))
            passed.append((score, candidate_id))
        else:
            negative.append({
                "schema_version": "1.0", "task_id": "24.4",
                "evidence_id": f"{execution_id}-{context.fold.fold_id}-{_safe_id(candidate_id)}-promotion-guard-failed",
                "reason": "promotion_guard_failed", "outer_fold_id": context.fold.fold_id,
                "candidate_id": candidate_id,
                "tested_configuration": candidate.model_dump(mode="json"),
                "guard_decision": record, "outer_metrics_used": False,
                "official_labels_used": False, "disposition": decision.disposition,
                "created_utc": _utc_now(),
            })
    ranked = sorted(passed, key=lambda item: (-item[0], item[1].encode("utf-8")))
    if ranked:
        selected_candidate = ranked[0][1]
        selected_epoch = int(final_results[selected_candidate]["selected_epoch"])
        selection_kind = "new_candidate_passed_all_inner_guards"
        fallback_reference = None
    else:
        selected_candidate = plan.fallback_candidate_id
        selected_epoch, fallback_hash = _fallback_epoch(context)
        selection_kind = "task_24_2_fallback_retained_no_new_candidate_passed"
        fallback_reference = {
            "decision_path": (BASELINE_DECISIONS_ROOT / f"{context.fold.fold_id}.json").relative_to(WORKSPACE).as_posix(),
            "decision_hash": fallback_hash,
            "source_candidate_id": "branch::rgb-pca-consistency",
        }
    selected = by_id[selected_candidate]
    decision = {
        "schema_version": "1.0", "task_id": "24.4",
        "decision_id": f"{execution_id}-{context.fold.fold_id}-final-inner-selection",
        "outer_fold_id": context.fold.fold_id, "held_out_variety": context.fold.validation_variety,
        "candidate_set": [candidate.model_dump(mode="json") for candidate in plan.candidates],
        "paired_screen_decision_hash": screen_decision["decision_hash"],
        "promotion_guard_decisions": guard_decisions,
        "eligible_candidates_ranked": [candidate_id for _score, candidate_id in ranked],
        "selected_candidate": selected_candidate, "selected_epoch": selected_epoch,
        "selected_target_routes": selected.target_routes,
        "selected_loss_key": selected.loss_key,
        "selected_spectral_normalization_key": selected.spectral_normalization_key,
        "selected_configuration_hash": selected.configuration_hash,
        "selection_kind": selection_kind, "fallback_reference": fallback_reference,
        "source_sample_ids": tuple(sorted(context.fold.train_ids, key=str.encode)),
        "source_sample_ids_hash": hash_object(tuple(sorted(context.fold.train_ids, key=str.encode))),
        "outer_test_overlap_count": 0, "metric_direction": MetricDirection.MAXIMIZE.value,
        "objective_metric": "mean_over_seeds.inner_pooled_oof.r2.Gsw",
        "tie_break_rule": plan.context.tie_break_rule,
        "epoch_tie_break_rule": plan.context.epoch_tie_break_rule,
        "seeds_used_for_selection": list(SEEDS),
        "outer_metrics_available_at_freeze": False,
        "task_14_13_metrics_used": False, "task_24_2_outer_metrics_used": False,
        "new_outer_metrics_used": False, "official_labels_used": False,
        "solution_csv_used": False, "official_budget_consumed": 0,
        "frozen_utc": _utc_now(),
    }
    decision["decision_hash"] = hash_object(decision)
    return decision, negative


def _verify_preregistration(plan: HyperLeafOptimizationPlan) -> dict[str, Any]:
    frozen = _read_json(PREREGISTRATION_PATH)
    observed_hashes = {candidate.candidate_id: candidate.configuration_hash for candidate in plan.candidates}
    checks = {
        "plan_hash": frozen["registry_hashes"]["plan_hash"] == plan.plan_hash,
        "candidate_hashes": frozen["candidate_configuration_hashes"] == observed_hashes,
        "comparison_context": frozen["comparison_identity"]["context_hash"] == plan.context.comparison_context_hash,
        "precision_profile": frozen["comparison_identity"]["precision_profile_config_hash"] == plan.context.precision_profile_config_hash,
        "candidate_count": len(plan.candidates) == 10,
        "stage_caps": tuple(stage.candidate_epoch_cap for stage in plan.screening_schedule) == (20, 80),
        "seals": all(value is False for value in frozen["seals"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"task-24.3 preregistration integrity failed: {checks}")
    return {"path": PREREGISTRATION_PATH.relative_to(WORKSPACE).as_posix(),
            "file_hash": hash_file(PREREGISTRATION_PATH), "checks": checks}


def execute_optimization(*, resume_execution_id: str | None = None) -> dict[str, Any]:
    if PureWindowsPath(str(WORKSPACE)).drive.casefold() != "g:" or PureWindowsPath(str(DATA_ROOT)).drive.casefold() != "g:":
        raise RuntimeError("task 24.4 is restricted to approved G: roots")
    template = load_remediation_template(PROFILE_PATH)
    if template.profile is not RemediationProfile.DETERMINISTIC_FLOAT32 or tuple(template.seed_set) != SEEDS:
        raise RuntimeError("task 24.4 requires the immutable deterministic-float32 profile")
    environment = _validate_environment(template)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    _set_determinism(SEEDS[0])
    plan = build_hyperleaf_optimization_plan(WORKSPACE)
    preregistration = _verify_preregistration(plan)

    manifest = _load_train_manifest()
    planner = ProtocolPlanner(FailureAtomicArtifactStore(OUTPUT_ROOT))
    protocol = plan_hyperleaf_lovo(planner, manifest, seed_set=SEEDS, publish=False)
    if protocol.protocol_hash != plan.context.protocol_hash or manifest.dataset_hash != plan.context.dataset_hash:
        raise RuntimeError("task-24.4 protocol or dataset identity differs from preregistration")
    contexts = build_nested_contexts(protocol, manifest)
    execution_id = resume_execution_id or datetime.now(timezone.utc).strftime("task-24-4-%Y%m%dT%H%M%S%fZ")
    execution_root = OUTPUT_ROOT / "executions" / execution_id
    report_path = execution_root / "task-24.4-report.json"
    if resume_execution_id:
        if not execution_root.is_dir():
            raise RuntimeError("requested task-24.4 execution does not exist")
        if report_path.exists():
            report = _read_json(report_path)
            return {"status": "complete", "execution_id": execution_id,
                    "report": report_path.relative_to(WORKSPACE).as_posix(),
                    "selected_candidate_by_outer": report["selected_candidate_by_outer"]}
    else:
        execution_root.mkdir(parents=True, exist_ok=False)

    created_utc = (
        _read_json(execution_root / "frozen-plan.json")["created_utc"]
        if resume_execution_id else _utc_now()
    )
    frozen_plan = {
        "schema_version": "1.0", "task_id": "24.4", "execution_id": execution_id,
        "created_utc": created_utc, "task_24_3_plan_hash": plan.plan_hash,
        "task_24_3_preregistration": preregistration,
        "comparison_context": plan.context.model_dump(mode="json"),
        "candidate_set": [candidate.model_dump(mode="json") for candidate in plan.candidates],
        "screening_schedule": [stage.model_dump(mode="json") for stage in plan.screening_schedule],
        "promotion_guard": plan.promotion_guard.model_dump(mode="json"),
        "training": {"optimizer": "AdamW", "learning_rate": LEARNING_RATE,
                     "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                     "evaluation_batch_size": EVALUATION_BATCH_SIZE,
                     "gradient_clip": GRADIENT_CLIP, "augmentation": STYLE_KEY,
                     "scheduler": "CosineAnnealingLR", "scheduler_t_max": PROMOTION_CAP},
        "scope": {"workspace": "G:/Datasets/HyperCross", "data_root": "G:/Datasets/HyperLeaf2024",
                  "device": "cuda:0", "process_count": 1, "numeric_precision": "float32",
                  "outer_evaluation_allowed": False, "official_images_allowed": False,
                  "official_labels_allowed": False, "solution_csv_allowed": False,
                  "task_14_13_metrics_allowed": False, "task_24_2_outer_metrics_allowed": False,
                  "official_budget": 0},
        "commands": {"execute": COMMAND_EXECUTE, "resume": COMMAND_RESUME.format(execution_id=execution_id)},
    }
    frozen_plan["frozen_plan_hash"] = hash_object(frozen_plan)
    if resume_execution_id:
        if _json_bytes(_read_json(execution_root / "frozen-plan.json")) != _json_bytes(frozen_plan):
            raise RuntimeError("resume request differs from immutable task-24.4 plan")
    else:
        _publish_json_once(execution_root / "frozen-plan.json", frozen_plan)
        _publish_json_once(execution_root / "environment-lock.json", environment.model_dump(mode="json"))
        _atomic_bytes(execution_root / "protocol.canonical.json", canonical_serialize(protocol))

    overall_started_ns = time.monotonic_ns()
    selected_by_outer: dict[str, str] = {}
    selected_epoch_by_outer: dict[str, int] = {}
    context_summaries: dict[str, Any] = {}
    all_resources: list[Mapping[str, Any]] = []
    for context in contexts:
        data = _load_context_data(
            manifest=manifest, protocol_hash=protocol.protocol_hash,
            context=context, execution_root=execution_root,
        )
        screen_results: dict[str, Mapping[str, Any]] = {}
        for candidate in plan.candidates:
            screen_results[candidate.candidate_id] = _train_to_cap(
                execution_id=execution_id, execution_root=execution_root, plan=plan,
                context=context, candidate=candidate, data=data, cap=SCREEN_CAP,
                stage="paired-screen-20-epochs", template=template,
                environment=environment, device=device,
            )
            all_resources.append(screen_results[candidate.candidate_id]["runtime_measurements"])
        screen_decision = _screen_decision(
            plan=plan, context=context, screen_results=screen_results, execution_id=execution_id
        )
        screen_path = execution_root / "decisions" / f"{context.fold.fold_id}-paired-screen.json"
        _publish_json_once(screen_path, screen_decision)
        for candidate in plan.candidates:
            if candidate.role is CandidateRole.OPTIMIZATION and candidate.candidate_id not in screen_decision["advanced_optimization_candidates"]:
                comparison = screen_decision["comparisons"][candidate.candidate_id]
                negative = {
                    "schema_version": "1.0", "task_id": "24.4",
                    "evidence_id": f"{execution_id}-{context.fold.fold_id}-{_safe_id(candidate.candidate_id)}-pruned-at-20",
                    "reason": "pruned_at_paired_screen_20_epochs",
                    "outer_fold_id": context.fold.fold_id, "candidate_id": candidate.candidate_id,
                    "tested_configuration": candidate.model_dump(mode="json"),
                    "matched_comparison": comparison,
                    "outer_metrics_used": False, "official_labels_used": False,
                    "disposition": "immutable_negative_evidence_no_outer_revival",
                    "created_utc": _utc_now(),
                }
                _append_once(execution_root / "negative-evidence.jsonl", negative)
                _append_once(OUTPUT_ROOT / "negative-evidence.jsonl", negative)

        final_results: dict[str, Mapping[str, Any]] = {}
        for candidate_id in screen_decision["advanced_candidates_and_guards"]:
            candidate = next(item for item in plan.candidates if item.candidate_id == candidate_id)
            final_results[candidate_id] = _train_to_cap(
                execution_id=execution_id, execution_root=execution_root, plan=plan,
                context=context, candidate=candidate, data=data, cap=PROMOTION_CAP,
                stage="promotion-screen-80-epochs", template=template,
                environment=environment, device=device,
            )
            all_resources.append(final_results[candidate_id]["runtime_measurements"])
        final_decision, negative_records = _final_decision(
            plan=plan, context=context, screen_decision=screen_decision,
            final_results=final_results, execution_id=execution_id,
        )
        for negative in negative_records:
            _append_once(execution_root / "negative-evidence.jsonl", negative)
            _append_once(OUTPUT_ROOT / "negative-evidence.jsonl", negative)
        final_path = execution_root / "decisions" / f"{context.fold.fold_id}.json"
        _publish_json_once(final_path, final_decision)
        comparison_path = execution_root / "comparisons" / f"{context.fold.fold_id}.json"
        _publish_json_once(comparison_path, {
            "schema_version": "1.0", "task_id": "24.4",
            "outer_fold_id": context.fold.fold_id,
            "paired_screen_comparisons": screen_decision["comparisons"],
            "promotion_guard_comparisons": final_decision["promotion_guard_decisions"],
            "outer_metrics_used": False, "official_labels_used": False,
        })
        selected_by_outer[context.fold.fold_id] = str(final_decision["selected_candidate"])
        selected_epoch_by_outer[context.fold.fold_id] = int(final_decision["selected_epoch"])
        context_summaries[context.fold.fold_id] = {
            "held_out_variety": context.fold.validation_variety,
            "paired_screen_decision": {"path": screen_path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(screen_path)},
            "final_selection_decision": {"path": final_path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(final_path)},
            "comparison": {"path": comparison_path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(comparison_path)},
            "advanced_optimization_candidates": screen_decision["advanced_optimization_candidates"],
            "advanced_candidates_and_guards": screen_decision["advanced_candidates_and_guards"],
            "selected_candidate": final_decision["selected_candidate"],
            "selected_epoch": final_decision["selected_epoch"],
            "selection_kind": final_decision["selection_kind"],
            "promotion_guard_decisions": final_decision["promotion_guard_decisions"],
        }
        del data
        torch.cuda.empty_cache()

    if len(selected_by_outer) != 4:
        raise RuntimeError("all four outer contexts require frozen inner-only decisions")
    report = {
        "schema_version": "1.0", "task_id": "24.4", "status": "complete",
        "execution_id": execution_id, "completed_utc": _utc_now(),
        "command": COMMAND_EXECUTE if not resume_execution_id else COMMAND_RESUME.format(execution_id=execution_id),
        "resume_command": COMMAND_RESUME.format(execution_id=execution_id),
        "frozen_plan": {"path": (execution_root / "frozen-plan.json").relative_to(WORKSPACE).as_posix(),
                        "hash": hash_file(execution_root / "frozen-plan.json")},
        "environment_lock": {"path": (execution_root / "environment-lock.json").relative_to(WORKSPACE).as_posix(),
                             "hash": hash_file(execution_root / "environment-lock.json")},
        "protocol": {"path": (execution_root / "protocol.canonical.json").relative_to(WORKSPACE).as_posix(),
                     "hash": hash_file(execution_root / "protocol.canonical.json"),
                     "dataset_hash": manifest.dataset_hash, "protocol_hash": protocol.protocol_hash},
        "contexts": context_summaries,
        "selected_candidate_by_outer": selected_by_outer,
        "selected_epoch_by_outer": selected_epoch_by_outer,
        "fallback_candidate_id": plan.fallback_candidate_id,
        "all_contexts_fell_back": all(value == plan.fallback_candidate_id for value in selected_by_outer.values()),
        "resource_summary": {
            "candidate_stage_records": len(all_resources),
            "wall_clock_duration_ms_sum": sum(int(item["wall_clock_duration_ms"]) for item in all_resources),
            "overall_invocation_wall_clock_duration_ms": int((time.monotonic_ns() - overall_started_ns) // 1_000_000),
            "peak_device_memory_bytes_max": max(int(item["peak_device_memory_bytes"]) for item in all_resources),
            "completed_training_steps_total": sum(int(item["completed_training_steps_total"]) for item in all_resources),
            "device_identity": environment.device_identity, "numeric_precision": "float32",
            "process_count": 1,
        },
        "seals": {"outer_evaluations_performed": 0, "official_budget_consumed": 0,
                  "official_images_read": False, "official_labels_read": False,
                  "solution_csv_read": False, "task_14_13_metrics_read": False,
                  "task_24_2_outer_metrics_read": False},
        "negative_evidence": {
            "execution_path": (execution_root / "negative-evidence.jsonl").relative_to(WORKSPACE).as_posix(),
            "global_path": (OUTPUT_ROOT / "negative-evidence.jsonl").relative_to(WORKSPACE).as_posix(),
        },
        "known_caveats": [
            "Task 24.4 contains nested-inner optimization evidence only and performs no held-out outer-variety evaluation.",
            "The candidate scheduler is frozen at an 80-epoch horizon; the paired screen is its exact epoch-20 boundary.",
            "Input volume is measured from context-scoped train.csv access and two reads of each permitted raw cube.",
        ],
    }
    report["report_hash"] = hash_object(report)
    _publish_json_once(report_path, report)
    summary = {
        "schema_version": "1.0", "task_id": "24.4", "status": "complete",
        "execution_id": execution_id, "report": report_path.relative_to(WORKSPACE).as_posix(),
        "report_hash": hash_file(report_path), "selected_candidate_by_outer": selected_by_outer,
        "selected_epoch_by_outer": selected_epoch_by_outer,
        "fallback_candidate_id": plan.fallback_candidate_id,
        "official_evaluation_budget_consumed": 0,
    }
    latest = OUTPUT_ROOT / "latest-task-24.4-summary.json"
    if latest.exists():
        previous = _read_json(latest)
        if previous.get("execution_id") != execution_id:
            raise RuntimeError("completed task-24.4 summary already exists under another append-only execution")
    else:
        _publish_json_once(latest, summary)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    _publish_json_once(ARTIFACT_ROOT / execution_id / "report-reference.json", {
        "schema_version": "1.0", "task_id": "24.4", "execution_id": execution_id,
        "report": report_path.relative_to(WORKSPACE).as_posix(), "report_hash": hash_file(report_path),
    })
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--resume", metavar="EXECUTION_ID")
    arguments = parser.parse_args(argv)
    try:
        execute_optimization(resume_execution_id=arguments.resume)
    except Exception as exc:
        print(json.dumps({"task_id": "24.4", "status": "failed", "error": repr(exc)}), file=sys.stderr, flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
