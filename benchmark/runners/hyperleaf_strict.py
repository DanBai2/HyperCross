"""Governed task-14.13 strict HyperLeaf multi-seed LOVO execution.

The executor consumes the immutable task-14.12 selection decision, freezes all
strict identities before training, evaluates only the selected candidate and
its declared controls, and never exposes an official-test or solution-label
path. All learned preprocessing is fitted on each outer training fold.
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
from torch import Tensor, nn

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS, HYPERLEAF_VARIETIES
from benchmark.catalog.raw_cube import open_tiff_cube
from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import PredictionArtifactWriter
from benchmark.models.hyperleaf_candidates import create_hyperleaf_candidate_registry
from benchmark.protocols.hyperleaf_lovo import plan_hyperleaf_lovo
from benchmark.protocols.planner import ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.reporting import ClaimEvidence, ClaimGate
from benchmark.schema import (
    FitBoundary,
    FitBoundaryScope,
    ImplementationKind,
    MetricDirection,
    PredictionKey,
    PredictionRecord,
    ResourceField,
    RuntimeMeasurements,
    SampleManifest,
    assess_resource_budget,
    canonical_serialize,
    hash_file,
    hash_object,
)
from benchmark.transforms.hyperleaf_candidates import (
    GswTargetTransform,
    HYPERLEAF_STYLE_CONFIGS,
    apply_hyperleaf_style,
)
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
    _append_jsonl,
    _atomic_bytes,
    _atomic_json,
    _atomic_torch,
    _forward,
    _load_train_manifest,
    _load_training_table,
    _ordered_oof,
    _predict_fold,
    _target_space,
    _tree_fingerprint,
    build_screening_candidates,
    compute_grouped_metrics,
)

WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "strict" / "hyperleaf"
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "strict.json"
SCREENING_SUMMARY = WORKSPACE / "runs_v2" / "screening" / "hyperleaf" / "latest-task-14.12-summary.json"
STRICT_SEEDS = (7, 42, 123)
SELECTED_EPOCH = 47
WINNER_ID = "branch::rgb-pca-consistency"
REQUIRED_GSW_CONTROL = "gsw::rgb-pca-consistency-0.05::mse::standardized"
COMMAND = "conda run --no-capture-output -n TCFM python -m benchmark.runners.hyperleaf_strict --execute"
INSTABILITY_STD_THRESHOLD = 0.05


@dataclass
class StrictFoldState:
    fold: ScreeningFold
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    completed_steps: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _canonical_json_content(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _publish_json_once(path: Path, value: Any) -> None:
    content = _canonical_json_content(value)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact already exists with different content: {path}")
        return
    _atomic_bytes(path, content)


def _resume_plan_matches(existing: Mapping[str, Any], regenerated: Mapping[str, Any]) -> bool:
    """Compare plans using the same JSON representation persisted at freeze time."""

    return _canonical_json_content(existing) == _canonical_json_content(regenerated)


def _verify_decision_hash(decision: Mapping[str, Any]) -> None:
    expected = str(decision.get("decision_hash", ""))
    payload = dict(decision)
    payload.pop("decision_hash", None)
    if expected != hash_object(payload):
        raise RuntimeError("task-14.12 frozen selection decision hash mismatch")


def resolve_strict_candidates(
    decision: Mapping[str, Any],
    declarations: Sequence[ScreeningCandidate] | None = None,
) -> tuple[ScreeningCandidate, ...]:
    """Resolve the frozen winner and controls without score-based reselection."""

    _verify_decision_hash(decision)
    frozen = decision.get("frozen_configuration")
    if not isinstance(frozen, dict):
        raise RuntimeError("task-14.12 decision lacks frozen_configuration")
    if decision.get("selected_candidate") != WINNER_ID or frozen.get("candidate_id") != WINNER_ID:
        raise RuntimeError("task-14.12 winner differs from the staged winner")
    if int(frozen.get("selected_epoch", -1)) != SELECTED_EPOCH:
        raise RuntimeError("task-14.12 selected epoch differs from frozen epoch 47")
    if decision.get("official_test_labels_accessible") is not False or int(decision.get("official_evaluation_budget_consumed", -1)) != 0:
        raise RuntimeError("task-14.12 official-label or official-budget guard is invalid")

    current = tuple(declarations or build_screening_candidates())
    by_id = {candidate.candidate_id: candidate for candidate in current}
    frozen_entries = decision.get("candidate_set")
    if not isinstance(frozen_entries, list):
        raise RuntimeError("task-14.12 decision lacks candidate_set")
    frozen_by_id = {str(item["candidate_id"]): item for item in frozen_entries}
    for candidate_id, item in frozen_by_id.items():
        if candidate_id not in by_id:
            raise RuntimeError(f"frozen candidate is absent from current declarations: {candidate_id}")
        if item.get("config_hash") != by_id[candidate_id].config_hash:
            raise RuntimeError(f"frozen candidate config hash mismatch: {candidate_id}")

    declared_controls = frozen.get("matched_controls")
    if not isinstance(declared_controls, list):
        raise RuntimeError("frozen winner lacks matched_controls")
    required_ids = {WINNER_ID, REQUIRED_GSW_CONTROL, *(str(value) for value in declared_controls)}
    missing = sorted(required_ids - set(by_id), key=str.encode)
    if missing:
        raise RuntimeError("required strict controls are undeclared: " + ", ".join(missing))
    return tuple(by_id[candidate_id] for candidate_id in sorted(required_ids, key=str.encode))


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


def _strict_folds(protocol: Any, manifest: SampleManifest) -> tuple[ScreeningFold, ...]:
    groups = {record.sample_id: record.group_id for record in manifest.records}
    folds = []
    observed: list[str] = []
    for outer in protocol.folds:
        varieties = {groups[sample_id] for sample_id in outer.outer_test_ids}
        if len(varieties) != 1:
            raise RuntimeError(f"outer fold does not hold out exactly one variety: {outer.fold_id}")
        variety = str(next(iter(varieties)))
        fold = ScreeningFold(
            fold_id=outer.fold_id,
            validation_variety=variety,
            train_ids=outer.outer_train_ids,
            validation_ids=outer.outer_test_ids,
        )
        if set(fold.train_ids) & set(fold.validation_ids):
            raise RuntimeError(f"outer fold train/test overlap: {fold.fold_id}")
        folds.append(fold)
        observed.extend(fold.validation_ids)
    if tuple(fold.validation_variety for fold in folds) != HYPERLEAF_VARIETIES:
        raise RuntimeError("LOVO outer fold order or variety coverage differs from frozen protocol")
    if len(observed) != len(set(observed)) or set(observed) != set(manifest.canonical_order):
        raise RuntimeError("LOVO outer folds do not cover each training sample exactly once")
    return tuple(folds)


def _fit_outer_preprocessing(
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
            scope=FitBoundaryScope.OUTER_TRAIN,
            allowed_fit_sample_ids=fold.train_ids,
            outer_test_sample_ids=fold.validation_ids,
        )
        target_artifacts[fold.fold_id] = manager.fit_target_normalization(
            targets[train_indices], fold.train_ids, boundary,
            dataset_hash=dataset_hash, protocol_hash=protocol_hash,
        )
        pca_artifacts[fold.fold_id] = manager.fit_pca(
            raw_spectra[train_indices], fold.train_ids, boundary,
            dataset_hash=dataset_hash, protocol_hash=protocol_hash,
            n_components=3, whiten=False,
        )
        if set(target_artifacts[fold.fold_id].fit_sample_ids) != set(fold.train_ids):
            raise RuntimeError("target normalization fit IDs differ from outer train")
        if set(pca_artifacts[fold.fold_id].fit_sample_ids) != set(fold.train_ids):
            raise RuntimeError("PCA fit IDs differ from outer train")
    return target_artifacts, pca_artifacts


def _resize_chw(values: np.ndarray, size: tuple[int, int] = (32, 64)) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False).squeeze(0).numpy().astype(np.float32, copy=False)


def _load_strict_data(
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
        path = DATA_ROOT / Path(records[sample_id].input_references["full_cube"])
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError("training cube left approved G: data root")
        input_volume += path.stat().st_size
        with open_tiff_cube(path, axis_order="BHW") as reader:
            raw = np.asarray(reader.memmap_view(), dtype=np.float64).mean(axis=(1, 2))
        if not np.isfinite(raw).all():
            raise RuntimeError(f"non-finite training cube: {sample_id}")
        raw_spectra.append(raw)
    raw_spectrum_array = np.stack(raw_spectra)
    target_artifacts, pca_artifacts = _fit_outer_preprocessing(
        folds, sample_ids, targets, raw_spectrum_array, manifest.dataset_hash, protocol_hash
    )
    for fold in folds:
        _publish_json_once(
            execution_root / "transforms" / fold.fold_id / "target-normalization.json",
            target_artifacts[fold.fold_id].model_dump(mode="json"),
        )
        _publish_json_once(
            execution_root / "transforms" / fold.fold_id / "pca.json",
            pca_artifacts[fold.fold_id].model_dump(mode="json"),
        )

    rgb_rows: list[np.ndarray] = []
    spectrum_rows: list[np.ndarray] = []
    pca_rows: dict[str, list[np.ndarray]] = {fold.fold_id: [] for fold in folds}
    manager = PreprocessingManager()
    for position, sample_id in enumerate(sample_ids):
        path = DATA_ROOT / Path(records[sample_id].input_references["full_cube"])
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
            for fold in folds:
                boundary = FitBoundary(
                    fold_id=fold.fold_id,
                    scope=FitBoundaryScope.OUTER_TRAIN,
                    allowed_fit_sample_ids=fold.train_ids,
                    outer_test_sample_ids=fold.validation_ids,
                )
                projected = manager.apply(pca_artifacts[fold.fold_id], pixels, boundary).reshape(reader.shape[1], reader.shape[2], 3)
                minimum = projected.min((0, 1), keepdims=True)
                maximum = projected.max((0, 1), keepdims=True)
                denominator = np.where(maximum - minimum > 1e-12, maximum - minimum, 1.0)
                pca_rows[fold.fold_id].append(_resize_chw(np.moveaxis(((projected - minimum) / denominator).astype(np.float32), -1, 0)))
        if (position + 1) % 100 == 0 or position + 1 == len(sample_ids):
            print(json.dumps({"stage": "strict_representation_materialization", "completed": position + 1, "total": len(sample_ids)}), flush=True)

    rgb_array = np.stack(rgb_rows).astype(np.float32)
    spectrum_array = np.stack(spectrum_rows).astype(np.float32)
    pca_arrays = {fold_id: np.stack(rows).astype(np.float32) for fold_id, rows in pca_rows.items()}
    if rgb_array.shape != (1590, 3, 32, 64) or spectrum_array.shape != (1590, 128):
        raise RuntimeError("strict representation shape is invalid")
    if any(values.shape != (1590, 3, 32, 64) for values in pca_arrays.values()):
        raise RuntimeError("strict fold PCA shape is invalid")
    return ScreeningData(
        sample_ids=tuple(sample_ids), id_to_index={sample_id: i for i, sample_id in enumerate(sample_ids)},
        varieties=tuple(str(records[sample_id].group_id) for sample_id in sample_ids), targets=targets,
        rgb=rgb_array, spectrum=spectrum_array, pca_by_fold=pca_arrays,
        target_artifacts=target_artifacts, pca_artifacts=pca_artifacts,
        input_volume_bytes=input_volume, train_csv_hash=hash_file(DATA_ROOT / "train.csv"),
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


def _build_state(candidate: ScreeningCandidate, fold: ScreeningFold, seed: int, device: torch.device) -> StrictFoldState:
    _set_determinism(seed)
    model = create_hyperleaf_candidate_registry().build(candidate.model_key)
    _replace_adaptive_pool(model)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SELECTED_EPOCH)
    return StrictFoldState(fold=fold, model=model, optimizer=optimizer, scheduler=scheduler)


def _task_loss(prediction: Tensor, targets: Tensor, candidate: ScreeningCandidate) -> Tensor:
    from benchmark.transforms.hyperleaf_candidates import GswLossKind, gsw_regression_loss
    other = (F.mse_loss(prediction[:, 0], targets[:, 0]) + F.mse_loss(prediction[:, 2], targets[:, 2])) / 2.0
    gsw = gsw_regression_loss(prediction[:, 1], targets[:, 1], loss_kind=GswLossKind(candidate.loss_kind))
    return (2.0 * other + gsw) / 3.0


def _train_epoch(state: StrictFoldState, candidate: ScreeningCandidate, data: ScreeningData, epoch: int, seed: int, device: torch.device) -> float:
    state.model.train()
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in state.fold.train_ids], dtype=np.int64)
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed * 100_000 + epoch)).numpy()
    losses = []
    style = HYPERLEAF_STYLE_CONFIGS[candidate.style_key]
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
        augmentation_seed = seed * 10_000_000 + epoch * 10_000 + batch_number
        rgb, styled_spectrum = apply_hyperleaf_style(rgb, spectrum, config=style, seed=augmentation_seed)
        pca, _ = apply_hyperleaf_style(pca, spectrum, config=style, seed=augmentation_seed)
        state.optimizer.zero_grad(set_to_none=True)
        prediction, auxiliary_loss = _forward(state, candidate, rgb, pca, styled_spectrum, targets=targets)
        loss = auxiliary_loss if candidate.model_key.endswith("rgb-pca-consistency.v1") else _task_loss(prediction, targets, candidate)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite strict training loss")
        loss.backward()
        if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in state.model.parameters()):
            raise RuntimeError("non-finite strict training gradient")
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), GRADIENT_CLIP)
        state.optimizer.step()
        state.completed_steps += 1
        losses.append(float(loss.detach().item()))
    state.scheduler.step()
    return float(np.mean(losses))


def _checkpoint_payload(state: StrictFoldState, candidate: ScreeningCandidate, seed: int, epoch: int, identity_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "candidate_id": candidate.candidate_id,
        "candidate_config_hash": candidate.config_hash, "frozen_identity_hash": identity_hash,
        "fold_id": state.fold.fold_id, "seed": seed, "completed_epoch": epoch,
        "completed_training_steps": state.completed_steps, "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(), "scheduler": state.scheduler.state_dict(),
        "stage": {"selected_epoch_policy": "fixed-task-14.12-epoch-47", "next_epoch": epoch + 1},
    }


def _resume_state(job_root: Path, state: StrictFoldState, candidate: ScreeningCandidate, seed: int, identity_hash: str, device: torch.device) -> tuple[int, list[dict[str, Any]]]:
    marker_path = job_root / "recovery" / "latest.json"
    if not marker_path.exists():
        return 0, []
    marker = _read_json(marker_path)
    epoch = int(marker["completed_epoch"])
    if marker.get("frozen_identity_hash") != identity_hash:
        raise RuntimeError("recovery frozen identity mismatch")
    checkpoint = torch.load(job_root / "recovery" / f"slot-{epoch % 2}.pt", map_location=device, weights_only=False)
    expected = (candidate.candidate_id, candidate.config_hash, state.fold.fold_id, seed, epoch, identity_hash)
    observed = (checkpoint["candidate_id"], checkpoint["candidate_config_hash"], checkpoint["fold_id"], checkpoint["seed"], checkpoint["completed_epoch"], checkpoint["frozen_identity_hash"])
    if observed != expected:
        raise RuntimeError("recovery checkpoint identity mismatch")
    state.model.load_state_dict(checkpoint["model"], strict=True)
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.scheduler.load_state_dict(checkpoint["scheduler"])
    state.completed_steps = int(checkpoint["completed_training_steps"])
    return epoch, list(marker.get("history", []))


def _job_identity(candidate: ScreeningCandidate, fold: ScreeningFold, seed: int, plan: Mapping[str, Any], environment_hash: str) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0", "task_id": "14.13", "dataset_hash": plan["dataset_hash"],
        "protocol_hash": plan["protocol_hash"], "model_key": candidate.model_key,
        "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
        "strict_config_hash": plan["strict_config_hash"], "route_hash": plan["route_hash"],
        "epoch_policy_hash": plan["epoch_policy_hash"], "preprocessing_hash": plan["preprocessing_hash"],
        "budget_hash": plan["budget_hash"], "fold_id": fold.fold_id, "seed": seed,
        "environment_lock_hash": environment_hash,
    }
    payload["frozen_identity_hash"] = hash_object(payload)
    return payload


def _run_job(
    execution_id: str, execution_root: Path, candidate: ScreeningCandidate, fold: ScreeningFold,
    seed: int, data: ScreeningData, plan: Mapping[str, Any], template: Any,
    environment: Any, device: torch.device, overall_started_ns: int,
) -> dict[str, Any]:
    safe_id = hashlib.sha256(candidate.candidate_id.encode("utf-8")).hexdigest()[:16]
    run_id = f"{execution_id}-{safe_id}-{fold.fold_id}-seed-{seed}"
    job_root = execution_root / "runs" / safe_id / f"seed-{seed}" / fold.fold_id
    result_path = job_root / "result.json"
    if result_path.exists():
        result = _read_json(result_path)
        if result.get("status") != "complete":
            raise RuntimeError(f"non-complete immutable result exists: {run_id}")
        return result
    identity = _job_identity(candidate, fold, seed, plan, environment.lock_hash)
    _publish_json_once(job_root / "frozen-identity.json", identity)
    identity_hash = str(identity["frozen_identity_hash"])
    state = _build_state(candidate, fold, seed, device)
    completed_epoch, history = _resume_state(job_root, state, candidate, seed, identity_hash, device)
    started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    for epoch in range(completed_epoch + 1, SELECTED_EPOCH + 1):
        loss = _train_epoch(state, candidate, data, epoch, seed, device)
        history.append({"epoch": epoch, "train_loss": loss, "completed_training_steps": state.completed_steps})
        payload = _checkpoint_payload(state, candidate, seed, epoch, identity_hash)
        _atomic_torch(job_root / "recovery" / f"slot-{epoch % 2}.pt", payload)
        _atomic_json(job_root / "recovery" / "latest.json", {
            "schema_version": "1.0", "frozen_identity_hash": identity_hash,
            "completed_epoch": epoch, "completed_training_steps": state.completed_steps, "history": history,
        })
        elapsed_ms = int((time.monotonic_ns() - started_ns) // 1_000_000)
        overall_ms = int((time.monotonic_ns() - overall_started_ns) // 1_000_000)
        measurements = RuntimeMeasurements(
            completed_epochs=epoch, completed_training_steps=state.completed_steps,
            wall_clock_duration_ms=elapsed_ms,
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
            device_identity=environment.device_identity, numeric_precision="float32",
            parameter_count=sum(parameter.numel() for parameter in state.model.parameters()),
            input_volume_bytes=data.input_volume_bytes,
        )
        assessment = assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch)
        if assessment.hard_exceeded or overall_ms > 259_200_000:
            raise RuntimeError(f"strict hard budget exceeded at epoch boundary {epoch}")
        print(json.dumps({"stage": "strict_training", "candidate_id": candidate.candidate_id, "seed": seed, "fold_id": fold.fold_id, "epoch": epoch, "target_epoch": SELECTED_EPOCH, "train_loss": loss}), flush=True)

    checkpoint = _checkpoint_payload(state, candidate, seed, SELECTED_EPOCH, identity_hash)
    _atomic_torch(job_root / "checkpoints" / "selected.pt", checkpoint)
    _atomic_torch(job_root / "checkpoints" / "last.pt", checkpoint)
    predictions = _predict_fold(state, candidate, data, device)
    indices = np.asarray([data.id_to_index[sample_id] for sample_id in fold.validation_ids])
    records = tuple(
        PredictionRecord(
            key=PredictionKey(sample_id=sample_id, fold_id=fold.fold_id, seed=seed, inference_branch=candidate.inference_branch),
            target_name="three-target-output", y_true=tuple(float(value) for value in data.targets[index]),
            y_pred=tuple(float(value) for value in prediction),
        )
        for sample_id, index, prediction in zip(fold.validation_ids, indices, predictions)
    )
    relative_prediction = (job_root / "predictions.canonical.json").relative_to(OUTPUT_ROOT).as_posix()
    published = PredictionArtifactWriter(FailureAtomicArtifactStore(OUTPUT_ROOT)).publish_shards(
        artifact_id=f"{run_id}-predictions", run_id=run_id,
        expected_sample_ids=fold.validation_ids, shards=(records,), relative_path=relative_prediction,
    )
    fold_metrics = compute_grouped_metrics(
        fold.validation_ids, (fold.fold_id,) * len(fold.validation_ids),
        data.targets[indices], predictions,
    )
    duration_ms = int((time.monotonic_ns() - started_ns) // 1_000_000)
    result = {
        "schema_version": "1.0", "task_id": "14.13", "status": "complete",
        "run_id": run_id, "candidate_id": candidate.candidate_id, "role": "winner" if candidate.candidate_id == WINNER_ID else "matched_control",
        "model_key": candidate.model_key, "candidate_config_hash": candidate.config_hash,
        "intentional_model_differences": {
            "route": candidate.inference_branch, "target_routes": list(candidate.target_routes),
            "style": candidate.style_key, "loss": candidate.loss_kind,
            "target_transform": candidate.target_transform, "consistency_weight": candidate.consistency_weight,
        },
        "fold_id": fold.fold_id, "held_out_variety": fold.validation_variety, "seed": seed,
        "selected_epoch": SELECTED_EPOCH, "metrics": fold_metrics,
        "prediction_artifact": published.reference.model_dump(mode="json"),
        "prediction_content_hash": published.artifact.content_hash,
        "checkpoints": {
            "selected": {"path": (job_root / "checkpoints" / "selected.pt").relative_to(WORKSPACE).as_posix(), "hash": hash_file(job_root / "checkpoints" / "selected.pt")},
            "last": {"path": (job_root / "checkpoints" / "last.pt").relative_to(WORKSPACE).as_posix(), "hash": hash_file(job_root / "checkpoints" / "last.pt")},
            "recovery": {"path": (job_root / "recovery" / f"slot-{SELECTED_EPOCH % 2}.pt").relative_to(WORKSPACE).as_posix(), "hash": hash_file(job_root / "recovery" / f"slot-{SELECTED_EPOCH % 2}.pt")},
        },
        "runtime_measurements": {
            "completed_epochs": SELECTED_EPOCH, "completed_training_steps": state.completed_steps,
            "wall_clock_duration_ms": duration_ms, "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "device_identity": environment.device_identity, "numeric_precision": "float32",
            "parameter_count": sum(parameter.numel() for parameter in state.model.parameters()),
            "input_volume_bytes": data.input_volume_bytes,
        },
        "official_evaluation_budget_consumed": 0, "completed_utc": _utc_now(),
    }
    _publish_json_once(result_path, result)
    return result


def _load_job_records(result: Mapping[str, Any]) -> tuple[PredictionRecord, ...]:
    relative = str(result["prediction_artifact"]["relative_path"])
    artifact = PredictionArtifactWriter.reload((OUTPUT_ROOT / Path(relative)).read_bytes())
    if artifact.content_hash != result["prediction_content_hash"]:
        raise RuntimeError("prediction artifact hash changed after publication")
    return artifact.records


def aggregate_candidate_metrics(seed_metrics: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate fixed-policy seed metrics with explicit seed dispersion."""

    if tuple(sorted(seed_metrics)) != STRICT_SEEDS:
        raise RuntimeError("aggregate requires exact seeds 7, 42, and 123")
    targets: dict[str, Any] = {}
    for target in HYPERLEAF_TARGETS:
        r2 = np.asarray([seed_metrics[seed]["pooled_oof"]["targets"][target]["r2"] for seed in STRICT_SEEDS], dtype=np.float64)
        mae = np.asarray([seed_metrics[seed]["pooled_oof"]["targets"][target]["mae"] for seed in STRICT_SEEDS], dtype=np.float64)
        targets[target] = {
            "pooled_oof_r2_seed_mean": float(r2.mean()), "pooled_oof_r2_seed_std": float(r2.std(ddof=1)),
            "pooled_oof_r2_seed_variance": float(r2.var(ddof=1)), "pooled_oof_r2_seed_min": float(r2.min()),
            "pooled_oof_r2_seed_max": float(r2.max()), "pooled_oof_mae_seed_mean": float(mae.mean()),
            "pooled_oof_mae_seed_std": float(mae.std(ddof=1)),
        }
    gsw = targets["Gsw"]
    return {
        "seed_set": list(STRICT_SEEDS), "aggregation_policy": "fixed per-seed pooled OOF and fold-macro",
        "targets": targets,
        "stability": {
            "metric": "pooled_oof.r2.Gsw", "seed_variance": gsw["pooled_oof_r2_seed_variance"],
            "seed_standard_deviation": gsw["pooled_oof_r2_seed_std"],
            "seed_range": gsw["pooled_oof_r2_seed_max"] - gsw["pooled_oof_r2_seed_min"],
            "instability_threshold_standard_deviation": INSTABILITY_STD_THRESHOLD,
            "unstable": gsw["pooled_oof_r2_seed_std"] > INSTABILITY_STD_THRESHOLD,
        },
    }


def _append_negative_both(execution_root: Path, record: Mapping[str, Any]) -> None:
    _append_jsonl(execution_root / "negative-evidence.jsonl", record)
    _append_jsonl(OUTPUT_ROOT / "negative-evidence.jsonl", record)


def execute_strict(*, resume_execution_id: str | None = None) -> dict[str, Any]:
    if PureWindowsPath(str(WORKSPACE)).drive.casefold() != "g:" or PureWindowsPath(str(DATA_ROOT)).drive.casefold() != "g:":
        raise RuntimeError("task 14.13 is restricted to approved G: roots")
    template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.STRICT or tuple(template.seed_set) != STRICT_SEEDS:
        raise RuntimeError("task 14.13 requires the immutable strict template and exact seed set")
    environment = validate_live_environment(template)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    overall_started_ns = time.monotonic_ns()

    summary = _read_json(SCREENING_SUMMARY)
    report_path = WORKSPACE / Path(str(summary["report"]))
    selection_path = report_path.parent / "frozen-selection-decision.json"
    screening_plan_path = report_path.parent / "frozen-screening-plan.json"
    decision = _read_json(selection_path)
    candidates = resolve_strict_candidates(decision)
    if summary.get("status") != "pass" or summary.get("selected_candidate") != WINNER_ID or int(summary.get("selected_epoch", -1)) != SELECTED_EPOCH:
        raise RuntimeError("candidate-screening checkpoint has not passed with the staged winner")

    manifest = _load_train_manifest()
    sample_ids, varieties, targets = _load_training_table(manifest)
    planner = ProtocolPlanner(FailureAtomicArtifactStore(OUTPUT_ROOT))
    protocol = plan_hyperleaf_lovo(planner, manifest, seed_set=STRICT_SEEDS, publish=False)
    folds = _strict_folds(protocol, manifest)
    if protocol.seed_set != STRICT_SEEDS or protocol.evaluation_budget.maximum_successful_evaluations != len(folds) * len(STRICT_SEEDS):
        raise RuntimeError("strict LOVO protocol budget or seeds differ from declaration")

    epoch_policy = {"kind": "fixed_from_task_14.12", "selected_epoch": SELECTED_EPOCH, "outer_label_reselection": False}
    preprocessing = {"target_normalization": "outer_train_only", "pca": "outer_train_only_full_cube", "fit_ids": "exact_outer_train_ids", "normalization_dtype": "float64_fit_float32_training"}
    candidate_table = [
        {
            **asdict(candidate), "config_hash": candidate.config_hash,
            "role": "winner" if candidate.candidate_id == WINNER_ID else "matched_control",
        }
        for candidate in candidates
    ]
    strict_config = {
        "candidate_ids": [candidate.candidate_id for candidate in candidates], "seed_set": list(STRICT_SEEDS),
        "fold_ids": [fold.fold_id for fold in folds], "epoch_policy": epoch_policy,
        "preprocessing": preprocessing, "metric_configuration": "hyperleaf-per-target-r2-mae-fold-macro-pooled-oof-v1",
        "device": "cuda:0", "process_count": 1, "precision": "float32", "deterministic": True,
    }
    execution_id = resume_execution_id or datetime.now(timezone.utc).strftime("task-14-13-%Y%m%dT%H%M%S%fZ")
    execution_root = OUTPUT_ROOT / "executions" / execution_id
    if resume_execution_id:
        if not execution_root.is_dir():
            raise RuntimeError("requested strict execution does not exist")
    else:
        execution_root.mkdir(parents=True, exist_ok=False)

    plan = {
        "schema_version": "1.0", "task_id": "14.13", "execution_id": execution_id,
        "created_utc": _utc_now() if not resume_execution_id else _read_json(execution_root / "frozen-strict-plan.json")["created_utc"],
        "dataset_hash": manifest.dataset_hash, "protocol_key": protocol.protocol_key,
        "protocol_hash": protocol.protocol_hash, "strict_config": strict_config,
        "strict_config_hash": hash_object(strict_config), "candidate_table": candidate_table,
        "candidate_space_hash": hash_object(candidate_table),
        "route_hash": hash_object([{"candidate_id": c.candidate_id, "route": c.inference_branch, "target_routes": c.target_routes, "consistency_weight": c.consistency_weight} for c in candidates]),
        "epoch_policy": epoch_policy, "epoch_policy_hash": hash_object(epoch_policy),
        "preprocessing": preprocessing, "preprocessing_hash": hash_object(preprocessing),
        "budget": template.resource_budget.model_dump(mode="json"), "budget_hash": hash_object(template.resource_budget.model_dump(mode="json")),
        "selection_source": {
            "summary": SCREENING_SUMMARY.relative_to(WORKSPACE).as_posix(), "summary_hash": hash_file(SCREENING_SUMMARY),
            "screening_plan": screening_plan_path.relative_to(WORKSPACE).as_posix(), "screening_plan_hash": hash_file(screening_plan_path),
            "selection_decision": selection_path.relative_to(WORKSPACE).as_posix(), "selection_decision_hash": hash_file(selection_path),
            "decision_hash": decision["decision_hash"], "winner": WINNER_ID, "selected_epoch": SELECTED_EPOCH,
        },
        "template": {"path": TEMPLATE_PATH.relative_to(WORKSPACE).as_posix(), "content_hash": hash_file(TEMPLATE_PATH), "template_hash": template.template_hash},
        "scope": {"write_root": "runs_v2/strict/hyperleaf", "official_test_images_allowed": False, "official_labels_allowed": False, "solution_csv_allowed": False, "official_evaluation_budget": 0},
        "freeze_verification": {"candidate": True, "config": True, "route": True, "epoch_policy": True, "preprocessing": True, "budget": True},
    }
    if resume_execution_id:
        existing_plan = _read_json(execution_root / "frozen-strict-plan.json")
        if not _resume_plan_matches(existing_plan, plan):
            raise RuntimeError("resume strict plan differs from frozen execution plan")
    else:
        _publish_json_once(execution_root / "frozen-strict-plan.json", plan)
        _publish_json_once(execution_root / "environment-lock.json", environment.model_dump(mode="json"))
        _atomic_bytes(execution_root / "protocol.canonical.json", canonical_serialize(protocol))

    readonly_before = {
        "src": _tree_fingerprint(WORKSPACE / "src"), "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "dataset": _tree_fingerprint(DATA_ROOT), "train_manifest": hash_file(TRAIN_MANIFEST),
        "screening_summary": hash_file(SCREENING_SUMMARY), "screening_plan": hash_file(screening_plan_path),
        "selection_decision": hash_file(selection_path),
    }
    data = _load_strict_data(manifest, sample_ids, folds, targets, protocol.protocol_hash, execution_root)
    all_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in candidates:
        for seed in STRICT_SEEDS:
            for fold in folds:
                try:
                    all_results.append(_run_job(execution_id, execution_root, candidate, fold, seed, data, plan, template, environment, device, overall_started_ns))
                except Exception as exc:
                    failure = {"candidate_id": candidate.candidate_id, "seed": seed, "fold_id": fold.fold_id, "error": repr(exc), "created_utc": _utc_now()}
                    failures.append(failure)
                    _append_negative_both(execution_root, {"schema_version": "1.0", "task_id": "14.13", "evidence_id": f"{execution_id}-{candidate.safe_id}-seed-{seed}-{fold.fold_id}-failed", "reason": "failed_or_unstable", "tested_configuration": asdict(candidate), "details": failure, "created_utc": _utc_now()})
                    print(json.dumps({"stage": "strict_job_failed", **failure}), file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"strict execution has {len(failures)} failed jobs; resume the execution after correction")

    candidate_summaries: dict[str, Any] = {}
    for candidate in candidates:
        seed_metrics: dict[int, Mapping[str, Any]] = {}
        seed_artifacts: dict[str, Any] = {}
        candidate_results = [result for result in all_results if result["candidate_id"] == candidate.candidate_id]
        for seed in STRICT_SEEDS:
            selected_results = [result for result in candidate_results if result["seed"] == seed]
            if len(selected_results) != len(folds):
                raise RuntimeError("strict candidate seed lacks complete folds")
            records = tuple(record for result in selected_results for record in _load_job_records(result))
            by_sample = {record.key.sample_id: record for record in records}
            if len(by_sample) != len(sample_ids) or set(by_sample) != set(sample_ids):
                raise RuntimeError("strict seed OOF prediction coverage is incomplete")
            ordered_records = tuple(by_sample[sample_id] for sample_id in sample_ids)
            truth = np.asarray([record.y_true for record in ordered_records], dtype=np.float64)
            prediction = np.asarray([record.y_pred for record in ordered_records], dtype=np.float64)
            fold_ids = tuple(record.key.fold_id for record in ordered_records)
            metrics = compute_grouped_metrics(sample_ids, fold_ids, truth, prediction)
            metrics["per_target"] = metrics["pooled_oof"]["targets"]
            seed_metrics[seed] = metrics
            payload = {
                "schema_version": "1.0", "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
                "seed": seed, "selected_epoch": SELECTED_EPOCH, "sample_count": len(sample_ids),
                "member_run_ids": [result["run_id"] for result in selected_results],
                "prediction_hashes": [result["prediction_content_hash"] for result in selected_results],
                "metrics": metrics,
                "records": [record.model_dump(mode="json") for record in ordered_records],
            }
            payload["aggregate_hash"] = hash_object(payload)
            seed_path = execution_root / "aggregates" / candidate.safe_id / f"seed-{seed}-oof.json"
            _publish_json_once(seed_path, payload)
            seed_artifacts[str(seed)] = {"path": seed_path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(seed_path)}
        aggregate = aggregate_candidate_metrics(seed_metrics)
        candidate_summaries[candidate.candidate_id] = {
            "role": "winner" if candidate.candidate_id == WINNER_ID else "matched_control",
            "configuration": {**asdict(candidate), "config_hash": candidate.config_hash},
            "per_seed_metrics": {str(seed): seed_metrics[seed] for seed in STRICT_SEEDS},
            "aggregate_metrics": aggregate, "seed_oof_artifacts": seed_artifacts,
            "member_run_ids": [result["run_id"] for result in candidate_results],
            "prediction_hashes": [result["prediction_content_hash"] for result in candidate_results],
        }

    winner_gsw = candidate_summaries[WINNER_ID]["aggregate_metrics"]["targets"]["Gsw"]["pooled_oof_r2_seed_mean"]
    control_scores = {candidate_id: summary["aggregate_metrics"]["targets"]["Gsw"]["pooled_oof_r2_seed_mean"] for candidate_id, summary in candidate_summaries.items() if candidate_id != WINNER_ID}
    best_control_id = max(control_scores, key=control_scores.get)
    if winner_gsw <= control_scores[best_control_id]:
        _append_negative_both(execution_root, {
            "schema_version": "1.0", "task_id": "14.13", "evidence_id": f"{execution_id}-winner-underperformance",
            "reason": "underperformance", "candidate_id": WINNER_ID,
            "tested_configuration": candidate_summaries[WINNER_ID]["configuration"],
            "candidate_metrics": candidate_summaries[WINNER_ID]["aggregate_metrics"],
            "details": {"control_id": best_control_id, "winner_gsw_r2_seed_mean": winner_gsw, "control_gsw_r2_seed_mean": control_scores[best_control_id]},
            "created_utc": _utc_now(),
        })
    if candidate_summaries[WINNER_ID]["aggregate_metrics"]["stability"]["unstable"]:
        _append_negative_both(execution_root, {
            "schema_version": "1.0", "task_id": "14.13", "evidence_id": f"{execution_id}-winner-seed-instability",
            "reason": "seed_instability", "candidate_id": WINNER_ID,
            "tested_configuration": candidate_summaries[WINNER_ID]["configuration"],
            "candidate_metrics": candidate_summaries[WINNER_ID]["aggregate_metrics"],
            "details": {"threshold": INSTABILITY_STD_THRESHOLD}, "created_utc": _utc_now(),
        })

    source_ids = set(str(value) for value in decision["source_sample_ids"])
    outer_ids = {sample_id for fold in folds for sample_id in fold.validation_ids}
    prior_selection_overlap = sorted(source_ids & outer_ids, key=str.encode)
    claim = ClaimGate.classify(ClaimEvidence(
        claim_id=f"{execution_id}-frozen-winner", implementation_kind=ImplementationKind.IN_REPO,
        detected_leakage=False, model_identity_mismatch=False, remote_sensing_sanity_track=False,
        random_split=False, outer_data_used_for_model_selection=bool(prior_selection_overlap), single_run=False,
        outer_data_used_for_ensemble_weight_search=False, outer_data_used_for_threshold_search=False,
        strict_protocol_complete=True, provenance_complete=True, seed_set_complete=True, folds_complete=True,
        matched_input_control_required=True, matched_input_control_complete=not bool(set(decision["frozen_configuration"]["matched_controls"]) - set(candidate_summaries)),
        prediction_artifact_complete=True, unified_runner_reproduced=True, score=winner_gsw,
    ))
    promotion = {
        "evaluator": "benchmark.reporting.ClaimGate", "frozen_winner": WINNER_ID,
        "winner_reselected_from_strict_outer_metrics": False,
        "decision": claim.model_dump(mode="json"),
        "prior_selection_source_outer_overlap_count": len(prior_selection_overlap),
        "prior_selection_overlap_hash": hash_object(prior_selection_overlap),
        "note": "The task-14.12 decision used all 1,590 training IDs, which overlap every task-14.13 outer test fold; the evaluator therefore applies the outer-data-selection ceiling.",
    }
    _publish_json_once(execution_root / "promotion-evaluation.json", promotion)

    readonly_after = {
        "src": _tree_fingerprint(WORKSPACE / "src"), "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "dataset": _tree_fingerprint(DATA_ROOT), "train_manifest": hash_file(TRAIN_MANIFEST),
        "screening_summary": hash_file(SCREENING_SUMMARY), "screening_plan": hash_file(screening_plan_path),
        "selection_decision": hash_file(selection_path),
    }
    if readonly_before != readonly_after:
        raise RuntimeError("a read-only source, dataset, or immutable selection artifact changed")
    elapsed_ms = int((time.monotonic_ns() - overall_started_ns) // 1_000_000)
    report = {
        "schema_version": "1.0", "task_id": "14.13", "status": "complete",
        "execution_id": execution_id, "command": COMMAND, "completed_utc": _utc_now(),
        "frozen_winner": WINNER_ID, "selected_epoch": SELECTED_EPOCH,
        "candidate_control_table": candidate_summaries, "promotion_evaluation": promotion,
        "identity": {
            "dataset_hash": manifest.dataset_hash, "protocol_hash": protocol.protocol_hash,
            "strict_config_hash": plan["strict_config_hash"], "candidate_space_hash": plan["candidate_space_hash"],
            "route_hash": plan["route_hash"], "epoch_policy_hash": plan["epoch_policy_hash"],
            "preprocessing_hash": plan["preprocessing_hash"], "budget_hash": plan["budget_hash"],
            "environment_lock_hash": environment.lock_hash, "runner_hash": hash_file(Path(__file__)),
            "model_source_hash": hash_file(WORKSPACE / "benchmark" / "models" / "hyperleaf_candidates.py"),
            "preprocessing_source_hash": hash_file(WORKSPACE / "benchmark" / "transforms" / "preprocessing.py"),
        },
        "resource_use": {
            "wall_clock_duration_ms": elapsed_ms,
            "peak_device_memory_bytes": max(int(result["runtime_measurements"]["peak_device_memory_bytes"]) for result in all_results),
            "completed_epochs_total": sum(int(result["runtime_measurements"]["completed_epochs"]) for result in all_results),
            "completed_training_steps_total": sum(int(result["runtime_measurements"]["completed_training_steps"]) for result in all_results),
            "input_volume_bytes": data.input_volume_bytes, "frozen_limits_per_run": template.resource_budget.model_dump(mode="json"),
        },
        "scope": {
            "process_count": 1, "device": "cuda:0", "numeric_precision": "float32", "deterministic": True,
            "outer_variety_exactly_once_per_seed": True, "outer_label_route_or_epoch_reselection": False,
            "official_test_images_read": False, "official_test_labels_read": False, "solution_csv_read": False,
            "official_evaluation_budget_consumed": 0, "official_one_shot_started": False,
            "read_only_inputs_unchanged": True,
        },
        "artifacts": {
            "frozen_plan": "frozen-strict-plan.json", "protocol": "protocol.canonical.json",
            "environment_lock": "environment-lock.json", "promotion_evaluation": "promotion-evaluation.json",
            "run_root": "runs/", "aggregate_root": "aggregates/", "negative_evidence": "negative-evidence.jsonl",
        },
        "caveats": [
            "The strict executor did not reselect candidates, routes, or epochs from task-14.13 outer labels.",
            "Task-14.12 selected the frozen winner and epoch using all training-partition varieties, so its selection source overlaps every task-14.13 outer test fold; ClaimGate limits this evidence to development.",
            "No HyperLeaf official-test image, official label, solution.csv value, or official evaluation was accessed.",
        ],
    }
    _publish_json_once(execution_root / "task-14.13-report.json", report)
    latest = {
        "status": "complete", "execution_id": execution_id,
        "report": (execution_root / "task-14.13-report.json").relative_to(WORKSPACE).as_posix(),
        "frozen_winner": WINNER_ID, "selected_epoch": SELECTED_EPOCH,
        "promotion_status": claim.status.value, "official_evaluation_budget_consumed": 0,
    }
    _atomic_json(OUTPUT_ROOT / "latest-task-14.13-summary.json", latest)
    return latest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute complete governed task-14.13 strict LOVO")
    parser.add_argument("--resume", metavar="EXECUTION_ID", help="resume one frozen task-14.13 execution")
    arguments = parser.parse_args(argv)
    if arguments.execute == bool(arguments.resume):
        parser.error("choose exactly one of --execute or --resume EXECUTION_ID")
    summary = execute_strict(resume_execution_id=arguments.resume)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
