"""Governed reproduction of the permitted HyperLeaf random-split baselines.

Task 14.10 deliberately uses only ``train.csv`` labels and read-only legacy
representations/checkpoints.  It never opens the official test partition or
``solution.csv``.  All newly produced evidence is written below
``runs_v2/baselines/hyperleaf``.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from benchmark.adapters.hyperleaf_classic import (
    HyperLeafClassicEstimatorAdapter,
    HyperLeafClassicKind,
    classic_model_key,
)
from benchmark.catalog.hyperleaf import GLOBAL_SPECTRUM_KEY, HYPERLEAF_TARGETS
from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import PredictionArtifactWriter, regression_metrics
from benchmark.models.checkpoint import ExternalCheckpointLoader
from benchmark.models.hyperleaf_baseline import (
    CONCAT_MODEL_KEY,
    SPECTRAL_MODEL_KEY,
    VISUAL_MODEL_KEY,
    create_hyperleaf_baseline_registry,
)
from benchmark.models.hyperleaf_route_a import (
    GLOBAL_FUSION_MODEL_KEY,
    create_hyperleaf_route_a_registry,
)
from benchmark.transforms.preprocessing import PreprocessingManager
from benchmark.recovery import FailureAtomicArtifactStore, RecoveryManager
from benchmark.registry import (
    ArtifactRecord,
    ClaimCeiling,
    ClaimDecision,
    FrozenRunIdentity,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.schema import (
    ClaimStatus,
    ExecutionMode,
    FitBoundary,
    FitBoundaryScope,
    FrozenRunConfig,
    ImplementationKind,
    InputContract,
    MetricSchema,
    NumericPrecision,
    PredictionKey,
    PredictionRecord,
    ResourceField,
    RunStatus,
    RuntimeMeasurements,
    TaskKind,
    assess_resource_budget,
    canonical_serialize,
    hash_file,
    hash_object,
)

WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
CACHE_ROOT = DATA_ROOT / "cache_routeA_leaf"
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "baseline.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "baselines" / "hyperleaf"
LEGACY_ROOT = WORKSPACE / "runs" / "leaf_20261005_200532"
FOLD_ID = "legacy-random80-20-seed42"
PROTOCOL_KEY = "leaf.random80-20.legacy-reproduction.v1"
SEED = 42
EPOCHS = 20
BATCH_SIZE = 64
LEGACY_TOLERANCE = 1e-6

LEGACY_MODELS: Mapping[str, tuple[str, str, str]] = {
    "visual": ("VisualOnly", "f2e5a1c282242946a2a42376c87e95f8ccb02381e0ee1f50b8dc8f3bb4fb4a1d", VISUAL_MODEL_KEY),
    "spectral": ("SpectralOnly", "69fa5036feba89debb4c5dab7b032795e3f80c3320e55f0286f34d8182339291", SPECTRAL_MODEL_KEY),
    "concat": ("ConcatModel", "6a1a1953fd0c87e5fb6f9bbe2e2419d8517b2a90d7dcafa6438892f7de8d0db1", CONCAT_MODEL_KEY),
    "routeA": ("CrossModalRouteA", "359cbdcd85613229b919d1611ec8e97b585daa2259d5721838555cf9773eaf4f", GLOBAL_FUSION_MODEL_KEY),
}


@dataclass(frozen=True)
class BaselineSplit:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class BaselineArrays:
    visible_rgb: np.ndarray
    global_spectrum: np.ndarray
    targets: np.ndarray
    sample_ids: tuple[str, ...]
    input_volume_bytes: int
    cache_hashes: Mapping[str, str]


def legacy_random_split(sample_ids: Sequence[str], seed: int = SEED) -> BaselineSplit:
    """Recreate the exact positional NumPy permutation used by the legacy runner."""

    if len(sample_ids) < 5 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("baseline split requires at least five unique sample IDs")
    permutation = np.random.default_rng(seed).permutation(len(sample_ids))
    cut = int(0.8 * len(sample_ids))
    train_indices = tuple(int(value) for value in permutation[:cut])
    validation_indices = tuple(int(value) for value in permutation[cut:])
    return BaselineSplit(
        train_ids=tuple(sample_ids[index] for index in train_indices),
        validation_ids=tuple(sample_ids[index] for index in validation_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def compare_metric_records(
    recorded: Mapping[str, Any], observed: Mapping[str, Any], tolerance: float = LEGACY_TOLERANCE
) -> dict[str, Any]:
    """Compare one observed legacy checkpoint metric record without rewriting it."""

    differences: dict[str, Any] = {}
    for field in ("loss", "r2_mean", "mae"):
        delta = abs(float(recorded[field]) - float(observed[field]))
        differences[field] = {"recorded": float(recorded[field]), "observed": float(observed[field]), "absolute_difference": delta, "within_tolerance": delta <= tolerance}
    per_target = []
    for target, expected, actual in zip(HYPERLEAF_TARGETS, recorded["r2"], observed["r2"]):
        delta = abs(float(expected) - float(actual))
        per_target.append({"target": target, "recorded": float(expected), "observed": float(actual), "absolute_difference": delta, "within_tolerance": delta <= tolerance})
    differences["r2"] = per_target
    differences["all_within_tolerance"] = all(
        item["within_tolerance"]
        for value in differences.values()
        for item in (value if isinstance(value, list) else [value])
        if isinstance(item, dict) and "within_tolerance" in item
    )
    differences["tolerance"] = tolerance
    return differences


def make_prediction_records(
    sample_ids: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray, *, branch: str
) -> tuple[PredictionRecord, ...]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    expected = (len(sample_ids), len(HYPERLEAF_TARGETS))
    if truth.shape != expected or prediction.shape != expected:
        raise ValueError(f"prediction arrays must have shape {expected}")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("prediction arrays must be finite")
    return tuple(
        PredictionRecord(
            key=PredictionKey(sample_id=sample_id, fold_id=FOLD_ID, seed=SEED, inference_branch=branch),
            target_name="three-target-output",
            y_true=tuple(float(value) for value in truth[index]),
            y_pred=tuple(float(value) for value in prediction[index]),
        )
        for index, sample_id in enumerate(sample_ids)
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _torch_bytes(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _tree_fingerprint(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().encode("utf-8")):
        stat = path.stat()
        entries.append({"path": path.relative_to(root).as_posix(), "file": path.is_file(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return hash_object(entries)


def _validate_roots() -> None:
    for path in (WORKSPACE, DATA_ROOT, CACHE_ROOT, LEGACY_ROOT):
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError(f"path must remain on G: {path}")
    if not CACHE_ROOT.is_dir() or not LEGACY_ROOT.is_dir():
        raise RuntimeError("required read-only legacy inputs are unavailable")


def _sample_id(value: Any) -> str:
    return f"{int(value):05d}"


def _load_arrays() -> tuple[BaselineArrays, pd.DataFrame]:
    csv_path = DATA_ROOT / "train.csv"
    frame = pd.read_csv(csv_path)
    if tuple(HYPERLEAF_TARGETS) != ("GrainWeight", "Gsw", "PhiPS2") or len(frame) != 1590:
        raise RuntimeError("unexpected HyperLeaf training table contract")
    sample_ids = tuple(_sample_id(value) for value in frame["ImageId"])
    visible, spectrum, hashes = [], [], {}
    volume = csv_path.stat().st_size
    for sample_id in sample_ids:
        path = CACHE_ROOT / f"{sample_id}.npz"
        hashes[sample_id] = hash_file(path)
        volume += path.stat().st_size
        with np.load(path, allow_pickle=False) as payload:
            visual = np.asarray(payload["v"], dtype=np.float32)
            spectral = np.asarray(payload["s"], dtype=np.float32)
        if visual.shape != (3, 32, 64) or spectral.shape != (128,):
            raise RuntimeError(f"legacy representation shape mismatch for {sample_id}")
        if not np.isfinite(visual).all() or not np.isfinite(spectral).all():
            raise RuntimeError(f"legacy representation is non-finite for {sample_id}")
        visible.append(visual)
        spectrum.append(spectral)
    targets = frame[list(HYPERLEAF_TARGETS)].to_numpy(dtype=np.float32)
    return BaselineArrays(
        visible_rgb=np.stack(visible),
        global_spectrum=np.stack(spectrum),
        targets=targets,
        sample_ids=sample_ids,
        input_volume_bytes=volume,
        cache_hashes=hashes,
    ), frame


def _legacy_model(class_name: str) -> torch.nn.Module:
    from src import hypercross_routeA as legacy

    cls = getattr(legacy, class_name)
    return cls("reg", 3) if class_name == "CrossModalRouteA" else cls(3)


def _legacy_metrics(y_true: Tensor, prediction: Tensor, batch_size: int = 128) -> dict[str, Any]:
    total = 0.0
    for start in range(0, len(y_true), batch_size):
        end = min(start + batch_size, len(y_true))
        total += float(F.mse_loss(prediction[start:end], y_true[start:end]).item()) * (end - start)
    mse = ((prediction - y_true) ** 2).mean(0)
    variance = ((y_true - y_true.mean(0)) ** 2).mean(0) + 1e-8
    r2 = 1 - mse / variance
    return {"loss": total / len(y_true), "r2_mean": float(r2.mean()), "r2": r2.tolist(), "mae": float((prediction - y_true).abs().mean())}


def _legacy_comparisons(
    arrays: BaselineArrays,
    split: BaselineSplit,
    device: torch.device,
    store: FailureAtomicArtifactStore,
    execution_id: str,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    train_targets = arrays.targets[np.asarray(split.train_indices)]
    mean = train_targets.mean(0)
    scale = train_targets.std(0) + 1e-6
    y_norm = torch.from_numpy((arrays.targets[np.asarray(split.validation_indices)] - mean) / scale).to(device)
    rgb = torch.from_numpy(arrays.visible_rgb[np.asarray(split.validation_indices)]).to(device)
    spectrum = torch.from_numpy(arrays.global_spectrum[np.asarray(split.validation_indices)]).to(device)
    loader = ExternalCheckpointLoader()
    writer = PredictionArtifactWriter(store)
    results: dict[str, Any] = {}
    references = []
    for name, (class_name, expected_hash, model_key) in LEGACY_MODELS.items():
        checkpoint = LEGACY_ROOT / f"{name}_best.pt"
        history_path = LEGACY_ROOT / f"{name}_history.json"
        before = hash_file(checkpoint)
        payload = loader.load(
            checkpoint,
            source_reference=f"legacy-g-drive://runs/leaf_20261005_200532/{checkpoint.name}",
            expected_content_hash=expected_hash,
            map_location=device,
        )
        model = _legacy_model(class_name).to(device=device, dtype=torch.float32)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        with torch.no_grad():
            prediction_norm = model(rgb, spectrum)
        observed = _legacy_metrics(y_norm, prediction_norm)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        recorded = max(history, key=lambda row: float(row["r2_mean"]))
        comparison = compare_metric_records(recorded, observed)
        raw_prediction = prediction_norm.detach().cpu().numpy().astype(np.float64) * scale + mean
        published = writer.publish_shards(
            artifact_id=f"{execution_id}-legacy-{name}-predictions",
            run_id=f"legacy-comparison-{execution_id}-{name}",
            expected_sample_ids=split.validation_ids,
            shards=(make_prediction_records(split.validation_ids, arrays.targets[np.asarray(split.validation_indices)], raw_prediction, branch=f"legacy-{name}-checkpoint"),),
            relative_path=f"executions/{execution_id}/legacy-comparison/{name}.predictions.canonical.json",
        )
        references.append(published.reference)
        results[name] = {
            "model_key": model_key,
            "checkpoint_reference": f"runs/leaf_20261005_200532/{checkpoint.name}",
            "checkpoint_sha256": before,
            "checkpoint_unchanged": before == hash_file(checkpoint),
            "history_reference": f"runs/leaf_20261005_200532/{history_path.name}",
            "history_sha256": hash_file(history_path),
            "recorded_best_epoch": int(recorded["epoch"]),
            "comparison": comparison,
            "prediction_artifact": published.reference.model_dump(mode="json"),
            "caveat": "legacy checkpoint preprocessing lineage is incomplete; comparison evidence is not formal v2 evidence",
        }
    return results, tuple(references)


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


def _replace_identity_adaptive_pools(module: torch.nn.Module) -> int:
    """Replace mathematically identity 16-to-16 adaptive pools with Identity.

    PyTorch lacks a deterministic CUDA backward for adaptive pooling even when
    the requested output length already equals the input length.  The legacy
    spectral encoder reaches length 16 before requesting length 16, so this
    replacement preserves values while satisfying the frozen deterministic
    execution policy.
    """

    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, torch.nn.AdaptiveAvgPool1d) and child.output_size == 16:
            setattr(module, name, torch.nn.Identity())
            replaced += 1
        else:
            replaced += _replace_identity_adaptive_pools(child)
    return replaced


def _training_loss(model: torch.nn.Module, prediction: Tensor, targets: Tensor, visual: Tensor, spectral: Tensor) -> Tensor:
    task_loss = F.mse_loss(prediction, targets)
    visual = F.normalize(visual, dim=1)
    spectral = F.normalize(spectral, dim=1)
    alignment = (1 - (visual * spectral).sum(1)).mean()
    legacy_model = getattr(model, "model", model)
    if hasattr(legacy_model, "temp"):
        temperature = legacy_model.temp.exp().clamp(0.03, 0.5)
    else:
        temperature = torch.tensor(0.07, device=targets.device)
    logits = visual @ spectral.t() / temperature
    labels = torch.arange(len(targets), device=targets.device)
    nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return task_loss + 0.08 * alignment + 0.04 * nce


def _metric_schema() -> MetricSchema:
    return MetricSchema(task_kind=TaskKind.MULTI_TARGET_REGRESSION, numeric_tolerance=1e-12, target_names=HYPERLEAF_TARGETS)


def _claim(model_kind: ImplementationKind) -> ClaimDecision:
    reasons = ["unified_task_14_10_runner", "train_labels_only", "legacy_random_split_reproduction"]
    blockers = ["random_split_development_protocol", "single_seed", "legacy_representation_cache_lineage_rebound_only"]
    return ClaimDecision(
        status=ClaimStatus.DEVELOPMENT,
        supporting_reasons=tuple(reasons),
        blocking_reasons=tuple(blockers),
        applied_ceilings=(ClaimCeiling(status=ClaimStatus.DEVELOPMENT, reason_code="random_split", evidence_reference=PROTOCOL_KEY),),
        prerequisite_checks={"official_labels_read": False, "unified_runner": True, "legacy_checkpoint_promoted": False, "single_process": True, "float32": True, "classic_model": model_kind is ImplementationKind.CLASSIC},
    )


def _register_artifact(registry: RunRegistry, reference: Any, kind: str, run_id: str | None) -> None:
    registry.append_artifact(ArtifactRecord(artifact=reference, artifact_kind=kind, run_id=run_id, created_utc=datetime.now(timezone.utc)))


def _publish_shared_metadata(
    store: FailureAtomicArtifactStore,
    registry: RunRegistry,
    execution_id: str,
    arrays: BaselineArrays,
    split: BaselineSplit,
    environment_lock: Any,
    template: Any,
) -> tuple[str, str, dict[str, Any]]:
    dataset_payload = {
        "schema_version": "1.0.0",
        "dataset_key": "hyperleaf2024",
        "partition": "train-only",
        "sample_ids": arrays.sample_ids,
        "train_csv_sha256": hash_file(DATA_ROOT / "train.csv"),
        "legacy_cache_source": "hyperleaf2024/cache_routeA_leaf",
        "cache_content_hashes": arrays.cache_hashes,
        "official_test_opened": False,
        "official_labels_read": False,
    }
    dataset_hash = hash_object(dataset_payload)
    split_payload = {
        "schema_version": "1.0.0",
        "protocol_key": PROTOCOL_KEY,
        "dataset_hash": dataset_hash,
        "fold_id": FOLD_ID,
        "train_ids": split.train_ids,
        "validation_ids": split.validation_ids,
        "split_algorithm": "numpy.default_rng(42).permutation; floor(0.8*N)",
        "legacy_epoch_policy": "select maximum validation mean R2 across 20 epochs",
        "evaluation_budget": {"official_maximum": 1, "official_consumed": 0},
    }
    protocol_hash = hash_object(split_payload)
    refs = {}
    for name, payload in (
        ("dataset", {**dataset_payload, "dataset_hash": dataset_hash}),
        ("protocol", {**split_payload, "protocol_hash": protocol_hash}),
        ("environment", environment_lock.model_dump(mode="json")),
        ("template-snapshot", template.model_dump(mode="json")),
    ):
        reference = store.publish_bytes(
            artifact_id=f"{execution_id}-{name}",
            relative_path=f"executions/{execution_id}/metadata/{name}.json",
            content=_json_bytes(payload),
            media_type="application/json",
        )
        _register_artifact(registry, reference, f"baseline_{name}", None)
        refs[name] = reference.model_dump(mode="json")
    return dataset_hash, protocol_hash, refs


def _frozen_config(template: Any, model_key: str, input_volume: int, model_parameters: Mapping[str, Any]) -> FrozenRunConfig:
    return FrozenRunConfig.create(
        model_key=model_key,
        protocol_key=PROTOCOL_KEY,
        execution_mode=ExecutionMode.DETERMINISTIC,
        numeric_precision=NumericPrecision.FLOAT32,
        framework_settings=template.framework_settings,
        input_volume_bytes=input_volume,
        checkpoint_retention=template.checkpoint_retention,
        resource_budget=template.resource_budget,
        parameters={
            "legacy_protocol_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "fold_id": FOLD_ID,
            "official_labels_allowed": False,
            **dict(model_parameters),
        },
    )


def _allocate(
    registry: RunRegistry,
    registration: Any,
    config: FrozenRunConfig,
    dataset_hash: str,
    protocol_hash: str,
    environment_hash: str,
    source_hash: str,
) -> tuple[RunRecord, FrozenRunIdentity]:
    identity = FrozenRunIdentity(
        dataset_hash=dataset_hash,
        protocol_hash=protocol_hash,
        model_key=registration.model_key,
        config_hash=config.config_hash,
        fold_id=FOLD_ID,
        seed=SEED,
        environment_lock_hash=environment_hash,
        environment_compatibility_policy="exact",
    )
    planned = registry.allocate_run(
        frozen_identity=identity,
        model_identity=registration.identity,
        input_contract=registration.input_contract,
        provenance=Provenance(
            code_revision=registration.source_revision,
            source_hash=source_hash,
            environment_hash=environment_hash,
            config_hash=config.config_hash,
            dataset_hash=dataset_hash,
            adaptation_notes=("task-14.10-governed-retraining", "read-only-legacy-cache-with-new-content-hash-binding"),
        ),
        resource_budget=config.resource_budget,
        claim_decision=_claim(registration.identity.implementation_kind),
    )
    registry.append_run(RunRecord.model_validate({**planned.model_dump(mode="python"), "status": RunStatus.RUNNING}))
    return planned, identity


def _publish_identity(store: FailureAtomicArtifactStore, registry: RunRegistry, execution_id: str, run_id: str, identity: FrozenRunIdentity) -> Any:
    reference = store.publish_bytes(
        artifact_id=f"{execution_id}-{run_id}-frozen-identity",
        relative_path=f"executions/{execution_id}/runs/{run_id}/frozen_run_identity.json",
        content=_json_bytes(identity.model_dump(mode="json")),
        media_type="application/json",
    )
    _register_artifact(registry, reference, "frozen_run_identity", run_id)
    return reference


def _publish_transform(store: FailureAtomicArtifactStore, registry: RunRegistry, execution_id: str, run_id: str, name: str, artifact: Any) -> Any:
    reference = store.publish_bytes(
        artifact_id=f"{execution_id}-{run_id}-{name}",
        relative_path=f"executions/{execution_id}/runs/{run_id}/transforms/{name}.canonical.json",
        content=canonical_serialize(artifact),
        media_type="application/json",
    )
    _register_artifact(registry, reference, f"baseline_{name}", run_id)
    return reference


def _publish_checkpoint(store: FailureAtomicArtifactStore, registry: RunRegistry, execution_id: str, run_id: str, role: str, payload: bytes, suffix: str) -> Any:
    reference = store.publish_bytes(
        artifact_id=f"{execution_id}-{run_id}-{role}-checkpoint",
        relative_path=f"executions/{execution_id}/runs/{run_id}/checkpoints/{role}.{suffix}",
        content=payload,
        media_type="application/octet-stream",
    )
    _register_artifact(registry, reference, f"baseline_{role}_checkpoint", run_id)
    return reference


def _budget_limit(config: FrozenRunConfig, field: ResourceField) -> int:
    return next(limit.value for limit in config.resource_budget.limits if limit.field == field.value)


def _complete_run(
    registry: RunRegistry,
    planned: RunRecord,
    *,
    transform_refs: Sequence[Any],
    selected_checkpoint: Any,
    recovery_ref: Any,
    prediction_ref: Any,
    metrics: Any,
    measurements: RuntimeMeasurements,
    caveats: Sequence[str],
    runner_metadata: Mapping[str, Any],
) -> None:
    running = registry.latest_run(planned.run_id)
    complete = RunRecord.model_validate({
        **running.model_dump(mode="python"),
        "status": RunStatus.COMPLETE,
        "transform_artifact_refs": tuple(transform_refs),
        "checkpoint_ref": selected_checkpoint,
        "recovery_artifact_ref": recovery_ref,
        "prediction_artifact_ref": prediction_ref,
        "metrics": metrics.model_dump(mode="json"),
        "runtime_measurements": measurements,
        "completed_utc": datetime.now(timezone.utc),
    })
    registry.append_run(complete)


def _train_neural(
    *,
    model_key: str,
    registration: Any,
    model: torch.nn.Module,
    arrays: BaselineArrays,
    split: BaselineSplit,
    target_artifact: Any,
    dataset_hash: str,
    protocol_hash: str,
    environment_lock: Any,
    template: Any,
    source_hash: str,
    store: FailureAtomicArtifactStore,
    registry: RunRegistry,
    recovery: RecoveryManager,
    execution_id: str,
    device: torch.device,
) -> dict[str, Any]:
    config = _frozen_config(template, model_key, arrays.input_volume_bytes, {"optimizer": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4, "alignment_weight": 0.08, "contrastive_weight": 0.04})
    planned, identity = _allocate(registry, registration, config, dataset_hash, protocol_hash, environment_lock.lock_hash, source_hash)
    identity_ref = _publish_identity(store, registry, execution_id, planned.run_id, identity)
    target_ref = _publish_transform(store, registry, execution_id, planned.run_id, "target-normalization", target_artifact)
    deterministic_pool_replacements = _replace_identity_adaptive_pools(model)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    train_index = np.asarray(split.train_indices)
    validation_index = np.asarray(split.validation_indices)
    mean = np.asarray(target_artifact.state["mean"], dtype=np.float32)
    scale = np.asarray(target_artifact.state["scale"], dtype=np.float32)
    normalized_targets = ((arrays.targets - mean) / scale).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(arrays.visible_rgb[train_index]), torch.from_numpy(arrays.global_spectrum[train_index]), torch.from_numpy(normalized_targets[train_index]))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    val_rgb = torch.from_numpy(arrays.visible_rgb[validation_index]).to(device)
    val_spectrum = torch.from_numpy(arrays.global_spectrum[validation_index]).to(device)
    val_targets = torch.from_numpy(normalized_targets[validation_index]).to(device)
    started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    best_score = -math.inf
    best_epoch = 0
    best_state = b""
    history = []
    latest_recovery = None
    steps = 0
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            for rgb, spectrum, targets in loader:
                rgb, spectrum, targets = rgb.to(device, non_blocking=True), spectrum.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction, visual, spectral = model(rgb, spectrum, return_aux=True)
                loss = _training_loss(model, prediction, targets, visual, spectral)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("non-finite neural baseline loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach().item()) * len(targets)
                seen += len(targets)
                steps += 1
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_prediction = model(val_rgb, val_spectrum)
            observed = _legacy_metrics(val_targets, val_prediction)
            history.append({"epoch": epoch, "train_loss": total_loss / seen, **observed})
            state = _torch_bytes(model.state_dict())
            if observed["r2_mean"] > best_score:
                best_score, best_epoch, best_state = observed["r2_mean"], epoch, state
            latest_recovery = recovery.save_checkpoint_boundary(
                run_id=planned.run_id,
                recovery_id=f"{execution_id}-{planned.run_id}-recovery-{epoch}",
                relative_path=f"executions/{execution_id}/runs/{planned.run_id}/recovery/epoch-{epoch}.canonical.json",
                model_state=state,
                optimizer_state=_torch_bytes(optimizer.state_dict()),
                scheduler_state=_torch_bytes(scheduler.state_dict()),
                stage_state=_json_bytes({"completed_epoch": epoch, "completed_training_steps": steps, "best_epoch": best_epoch, "best_score": best_score}),
                last_completed_stage=f"epoch_{epoch}",
                completed_checkpoint_boundary=epoch,
            )
            interim = RuntimeMeasurements(
                completed_epochs=epoch,
                completed_training_steps=steps,
                wall_clock_duration_ms=(time.monotonic_ns() - started_ns) // 1_000_000,
                peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
                device_identity=environment_lock.device_identity,
                numeric_precision="float32",
                parameter_count=registration.identity.parameter_count,
                input_volume_bytes=arrays.input_volume_bytes,
            )
            if assess_resource_budget(config.resource_budget, interim, checkpoint_boundary=epoch).hard_exceeded:
                raise RuntimeError("hard resource limit reached at neural checkpoint boundary")
    except Exception:
        recovery.record_interruption(planned.run_id, status=RunStatus.FAILED, last_completed_stage=f"epoch_{len(history)}")
        raise
    last_state = _torch_bytes(model.state_dict())
    selected_ref = _publish_checkpoint(store, registry, execution_id, planned.run_id, "selected", best_state, "pt")
    last_ref = _publish_checkpoint(store, registry, execution_id, planned.run_id, "last", last_state, "pt")
    reloaded = type(model)().to(device=device, dtype=torch.float32)
    reloaded.load_state_dict(torch.load(io.BytesIO(best_state), map_location=device, weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        selected_normalized = reloaded(val_rgb, val_spectrum)
    raw_prediction = selected_normalized.detach().cpu().numpy().astype(np.float64) * scale + mean
    records = make_prediction_records(split.validation_ids, arrays.targets[validation_index], raw_prediction, branch=registration.model_key)
    published = PredictionArtifactWriter(store).publish_shards(
        artifact_id=f"{execution_id}-{planned.run_id}-predictions",
        run_id=planned.run_id,
        expected_sample_ids=split.validation_ids,
        shards=(records,),
        relative_path=f"executions/{execution_id}/runs/{planned.run_id}/predictions.canonical.json",
    )
    _register_artifact(registry, published.reference, "baseline_prediction", planned.run_id)
    metrics = regression_metrics(published.artifact, _metric_schema())
    elapsed = (time.monotonic_ns() - started_ns) // 1_000_000
    measurements = RuntimeMeasurements(
        completed_epochs=EPOCHS,
        completed_training_steps=steps,
        wall_clock_duration_ms=elapsed,
        peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
        device_identity=environment_lock.device_identity,
        numeric_precision="float32",
        parameter_count=registration.identity.parameter_count,
        input_volume_bytes=arrays.input_volume_bytes,
    )
    assessment = assess_resource_budget(config.resource_budget, measurements, checkpoint_boundary=EPOCHS)
    _complete_run(
        registry,
        planned,
        transform_refs=(target_ref,),
        selected_checkpoint=selected_ref,
        recovery_ref=latest_recovery.artifact,
        prediction_ref=published.reference,
        metrics=metrics,
        measurements=measurements,
        caveats=("development-only random split", "legacy cache lineage was incomplete and is newly hash-bound, not retroactively repaired", "single seed 42"),
        runner_metadata={"identity_artifact": identity_ref.model_dump(mode="json"), "selected_epoch": best_epoch, "selected_normalized_mean_r2": best_score, "last_checkpoint": last_ref.model_dump(mode="json"), "history": history, "budget_assessment": assessment.model_dump(mode="json")},
    )
    return {"run_id": planned.run_id, "model_key": model_key, "selected_epoch": best_epoch, "normalized_selection_mean_r2": best_score, "deterministic_identity_pool_replacements": deterministic_pool_replacements, "metrics": metrics.model_dump(mode="json"), "prediction_artifact": published.reference.model_dump(mode="json"), "selected_checkpoint": selected_ref.model_dump(mode="json"), "last_checkpoint": last_ref.model_dump(mode="json"), "recovery_artifact": latest_recovery.artifact.model_dump(mode="json"), "runtime_measurements": measurements.model_dump(mode="json"), "budget_assessment": assessment.model_dump(mode="json")}


def _train_classic(
    *,
    kind: HyperLeafClassicKind,
    arrays: BaselineArrays,
    split: BaselineSplit,
    target_artifact: Any,
    dataset_hash: str,
    protocol_hash: str,
    environment_lock: Any,
    template: Any,
    source_hash: str,
    store: FailureAtomicArtifactStore,
    registry: RunRegistry,
    recovery: RecoveryManager,
    execution_id: str,
) -> dict[str, Any]:
    model_key = classic_model_key(kind, GLOBAL_SPECTRUM_KEY)
    parameters = {"n_components": 8} if kind is HyperLeafClassicKind.PLSR else {"n_estimators": 256, "random_state": SEED, "n_jobs": 1}
    estimator = HyperLeafClassicEstimatorAdapter(model_key=model_key, kind=kind, seed=SEED, parameters=parameters)
    contract = InputContract(
        representation_key=GLOBAL_SPECTRUM_KEY,
        modalities=("global_spectrum",),
        tensor_shapes={"global_spectrum": (128,)},
        dtypes={"global_spectrum": "float32"},
        channel_or_band_order={"global_spectrum": ("wavelength_ascending",)},
        preprocessing_key="hyperleaf.classic.fold-impute-band-target.v1",
    )
    from benchmark.schema import ModelIdentity
    identity_model = ModelIdentity(model_key=model_key, display_name=f"HyperLeaf {kind.value.upper()} global-spectrum baseline", class_fully_qualified_name=f"{HyperLeafClassicEstimatorAdapter.__module__}.{HyperLeafClassicEstimatorAdapter.__qualname__}", implementation_kind=ImplementationKind.CLASSIC, parameter_count=0, source_revision="hyperleaf-sklearn-adapter.v1")
    registration = type("Registration", (), {"model_key": model_key, "identity": identity_model, "input_contract": contract, "source_revision": "hyperleaf-sklearn-adapter.v1"})()
    config = _frozen_config(template, model_key, arrays.input_volume_bytes, parameters)
    planned, frozen_identity = _allocate(registry, registration, config, dataset_hash, protocol_hash, environment_lock.lock_hash, source_hash)
    identity_ref = _publish_identity(store, registry, execution_id, planned.run_id, frozen_identity)
    boundary = FitBoundary(fold_id=FOLD_ID, scope=FitBoundaryScope.OUTER_TRAIN, allowed_fit_sample_ids=split.train_ids, outer_test_sample_ids=split.validation_ids)
    preprocessing = PreprocessingManager()
    train_index = np.asarray(split.train_indices)
    val_index = np.asarray(split.validation_indices)
    raw_train = arrays.global_spectrum[train_index].astype(np.float64)
    imputation = preprocessing.fit_imputation(raw_train, split.train_ids, boundary, dataset_hash=dataset_hash, protocol_hash=protocol_hash)
    imputed = preprocessing.apply(imputation, raw_train, boundary)
    band = preprocessing.fit_band_normalization(imputed, split.train_ids, boundary, dataset_hash=dataset_hash, protocol_hash=protocol_hash)
    x_train = preprocessing.apply(band, imputed, boundary)
    y_train = preprocessing.apply(target_artifact, arrays.targets[train_index], boundary)
    transform_refs = (
        _publish_transform(store, registry, execution_id, planned.run_id, "imputation", imputation),
        _publish_transform(store, registry, execution_id, planned.run_id, "band-normalization", band),
        _publish_transform(store, registry, execution_id, planned.run_id, "target-normalization", target_artifact),
    )
    started_ns = time.monotonic_ns()
    try:
        estimator.fit(x_train, y_train)
        model_bytes = pickle.dumps(estimator, protocol=5)
        recovery_artifact = recovery.save_checkpoint_boundary(
            run_id=planned.run_id,
            recovery_id=f"{execution_id}-{planned.run_id}-recovery-1",
            relative_path=f"executions/{execution_id}/runs/{planned.run_id}/recovery/fit-boundary.canonical.json",
            model_state=model_bytes,
            optimizer_state=_json_bytes({"not_applicable": True}),
            scheduler_state=_json_bytes({"not_applicable": True}),
            stage_state=_json_bytes({"completed_fit": True, "completed_training_steps": 1}),
            last_completed_stage="classic_fit",
            completed_checkpoint_boundary=1,
        )
    except Exception:
        recovery.record_interruption(planned.run_id, status=RunStatus.FAILED, last_completed_stage="classic_fit_failed")
        raise
    selected_ref = _publish_checkpoint(store, registry, execution_id, planned.run_id, "selected", model_bytes, "pkl")
    last_ref = _publish_checkpoint(store, registry, execution_id, planned.run_id, "last", model_bytes, "pkl")
    restored = pickle.loads(model_bytes)
    raw_val = arrays.global_spectrum[val_index].astype(np.float64)
    x_val = preprocessing.apply(band, preprocessing.apply(imputation, raw_val, boundary), boundary)
    normalized_prediction = restored.predict(x_val)
    mean = np.asarray(target_artifact.state["mean"], dtype=np.float64)
    scale = np.asarray(target_artifact.state["scale"], dtype=np.float64)
    raw_prediction = normalized_prediction * scale + mean
    records = make_prediction_records(split.validation_ids, arrays.targets[val_index], raw_prediction, branch=model_key)
    published = PredictionArtifactWriter(store).publish_shards(
        artifact_id=f"{execution_id}-{planned.run_id}-predictions",
        run_id=planned.run_id,
        expected_sample_ids=split.validation_ids,
        shards=(records,),
        relative_path=f"executions/{execution_id}/runs/{planned.run_id}/predictions.canonical.json",
    )
    _register_artifact(registry, published.reference, "baseline_prediction", planned.run_id)
    metrics = regression_metrics(published.artifact, _metric_schema())
    measurements = RuntimeMeasurements(
        completed_epochs=1,
        completed_training_steps=1,
        wall_clock_duration_ms=(time.monotonic_ns() - started_ns) // 1_000_000,
        peak_device_memory_bytes=0,
        device_identity=environment_lock.device_identity,
        numeric_precision="float32-input/float64-sklearn",
        parameter_count=0,
        input_volume_bytes=arrays.input_volume_bytes,
    )
    assessment = assess_resource_budget(config.resource_budget, measurements, checkpoint_boundary=1)
    _complete_run(
        registry,
        planned,
        transform_refs=transform_refs,
        selected_checkpoint=selected_ref,
        recovery_ref=recovery_artifact.artifact,
        prediction_ref=published.reference,
        metrics=metrics,
        measurements=measurements,
        caveats=("development-only random split", "classic estimator executes on CPU by design", "legacy cache lineage was incomplete and is newly hash-bound, not retroactively repaired", "single seed 42"),
        runner_metadata={"identity_artifact": identity_ref.model_dump(mode="json"), "last_checkpoint": last_ref.model_dump(mode="json"), "budget_assessment": assessment.model_dump(mode="json")},
    )
    return {"run_id": planned.run_id, "model_key": model_key, "metrics": metrics.model_dump(mode="json"), "prediction_artifact": published.reference.model_dump(mode="json"), "selected_checkpoint": selected_ref.model_dump(mode="json"), "last_checkpoint": last_ref.model_dump(mode="json"), "recovery_artifact": recovery_artifact.artifact.model_dump(mode="json"), "runtime_measurements": measurements.model_dump(mode="json"), "budget_assessment": assessment.model_dump(mode="json")}


def _close_orphaned_runs(registry: RunRegistry, recovery: RecoveryManager) -> tuple[str, ...]:
    """Append failure snapshots for active records left by a prior terminated process."""

    if not registry.path.exists():
        return ()
    latest: dict[str, dict[str, Any]] = {}
    for line in registry.path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("entry_kind") == "run":
            record = entry["record"]
            latest[str(record["run_id"])] = record
    orphaned = tuple(
        sorted(
            (run_id for run_id, record in latest.items() if record.get("status") == RunStatus.RUNNING.value),
            key=lambda value: value.encode("utf-8"),
        )
    )
    for run_id in orphaned:
        recovery.record_interruption(
            run_id,
            status=RunStatus.FAILED,
            last_completed_stage="prior_process_terminated_after_last_recovery_boundary",
        )
    return orphaned


def execute_baseline() -> dict[str, Any]:
    _validate_roots()
    template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.BASELINE:
        raise RuntimeError("task 14.10 requires the immutable baseline template")
    environment_lock = validate_live_environment(template)
    _set_determinism()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    execution_id = datetime.now(timezone.utc).strftime("task-14-10-%Y%m%dT%H%M%S%fZ")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    store = FailureAtomicArtifactStore(OUTPUT_ROOT)
    registry = RunRegistry(OUTPUT_ROOT / "registry.jsonl")
    recovery = RecoveryManager(registry, store)
    orphaned_run_ids = _close_orphaned_runs(registry, recovery)
    readonly_before = {name: _tree_fingerprint(WORKSPACE / name) for name in ("src", "runs")}
    arrays, _ = _load_arrays()
    split = legacy_random_split(arrays.sample_ids)
    if len(split.train_ids) != 1272 or len(split.validation_ids) != 318 or set(split.train_ids) & set(split.validation_ids):
        raise RuntimeError("legacy split recreation failed")
    dataset_hash, protocol_hash, metadata_refs = _publish_shared_metadata(store, registry, execution_id, arrays, split, environment_lock, template)
    boundary = FitBoundary(fold_id=FOLD_ID, scope=FitBoundaryScope.OUTER_TRAIN, allowed_fit_sample_ids=split.train_ids, outer_test_sample_ids=split.validation_ids)
    target_artifact = PreprocessingManager().fit_target_normalization(arrays.targets[np.asarray(split.train_indices)], split.train_ids, boundary, dataset_hash=dataset_hash, protocol_hash=protocol_hash)
    legacy_results, legacy_prediction_refs = _legacy_comparisons(arrays, split, device, store, execution_id)

    route_registry = create_hyperleaf_route_a_registry()
    control_registry = create_hyperleaf_baseline_registry()
    neural_specs = (
        (VISUAL_MODEL_KEY, control_registry.resolve(VISUAL_MODEL_KEY), control_registry.build(VISUAL_MODEL_KEY)),
        (SPECTRAL_MODEL_KEY, control_registry.resolve(SPECTRAL_MODEL_KEY), control_registry.build(SPECTRAL_MODEL_KEY)),
        (CONCAT_MODEL_KEY, control_registry.resolve(CONCAT_MODEL_KEY), control_registry.build(CONCAT_MODEL_KEY)),
        (GLOBAL_FUSION_MODEL_KEY, route_registry.resolve(GLOBAL_FUSION_MODEL_KEY), route_registry.build(GLOBAL_FUSION_MODEL_KEY)),
    )
    neural_source_hash = hash_object({"runner": hash_file(Path(__file__)), "baseline_adapters": hash_file(WORKSPACE / "benchmark" / "models" / "hyperleaf_baseline.py"), "route_adapter": hash_file(WORKSPACE / "benchmark" / "models" / "hyperleaf_route_a.py"), "legacy_model_source": hash_file(WORKSPACE / "src" / "hypercross_routeA.py")})
    classic_source_hash = hash_object({"runner": hash_file(Path(__file__)), "classic_adapter": hash_file(WORKSPACE / "benchmark" / "adapters" / "hyperleaf_classic.py"), "sklearn_version": environment_lock.sklearn_version})
    runs = []
    for model_key, registration, model in neural_specs:
        runs.append(_train_neural(model_key=model_key, registration=registration, model=model, arrays=arrays, split=split, target_artifact=target_artifact, dataset_hash=dataset_hash, protocol_hash=protocol_hash, environment_lock=environment_lock, template=template, source_hash=neural_source_hash, store=store, registry=registry, recovery=recovery, execution_id=execution_id, device=device))
    for kind in (HyperLeafClassicKind.PLSR, HyperLeafClassicKind.RANDOM_FOREST):
        runs.append(_train_classic(kind=kind, arrays=arrays, split=split, target_artifact=target_artifact, dataset_hash=dataset_hash, protocol_hash=protocol_hash, environment_lock=environment_lock, template=template, source_hash=classic_source_hash, store=store, registry=registry, recovery=recovery, execution_id=execution_id))

    readonly_after = {name: _tree_fingerprint(WORKSPACE / name) for name in ("src", "runs")}
    readonly_unchanged = readonly_before == readonly_after
    if not readonly_unchanged:
        raise RuntimeError("read-only src or runs changed during task 14.10")
    report = {
        "schema_version": "1.0.0",
        "task_id": "14.10",
        "status": "pass",
        "execution_id": execution_id,
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": "conda run --no-capture-output -n TCFM python -m benchmark.runners.hyperleaf_baseline --execute",
        "scope": {"dataset_key": "hyperleaf2024", "partition": "train-only", "output_root": "runs_v2/baselines/hyperleaf", "src_and_runs_read_only": readonly_unchanged, "official_test_images_read": False, "official_solution_csv_read": False, "official_labels_read": False, "official_evaluation_budget_consumed": 0, "process_count": 1},
        "environment": environment_lock.model_dump(mode="json"),
        "template": {"reference": "configs/fast_lane/baseline.json", "content_hash": hash_file(TEMPLATE_PATH), "template_hash": template.template_hash, "numeric_precision": template.numeric_precision.value, "execution_mode": template.execution_mode.value, "framework_settings": template.framework_settings.model_dump(mode="json"), "resource_budget": template.resource_budget.model_dump(mode="json")},
        "identity": {"dataset_hash": dataset_hash, "protocol_hash": protocol_hash, "protocol_key": PROTOCOL_KEY, "fold_id": FOLD_ID, "model_source_hash": neural_source_hash, "classic_source_hash": classic_source_hash, "metadata_artifacts": metadata_refs},
        "split": {"train_count": len(split.train_ids), "validation_count": len(split.validation_ids), "seed": SEED, "epochs": EPOCHS},
        "legacy_tolerance_comparisons": legacy_results,
        "legacy_prediction_artifacts": [reference.model_dump(mode="json") for reference in legacy_prediction_refs],
        "prior_orphaned_runs_closed_as_failed": list(orphaned_run_ids),
        "governed_runs": runs,
        "caveats": [
            "All results use the historical random 80/20 development protocol and cannot support strict claims.",
            "Legacy checkpoint preprocessing lineage is incomplete; legacy reloads are tolerance checks only and were not promoted to formal v2 evidence.",
            "Governed retraining binds the exact read-only legacy representation cache bytes and transform state into new artifacts; it does not retroactively establish the cache's original creation lineage.",
            "Only seed 42 was executed, as frozen by the baseline template.",
            "PLSR and RF execute on CPU inside the TCFM process; the neural baselines execute on cuda:0 RTX 5080.",
            "For deterministic CUDA training, mathematically identity AdaptiveAvgPool1d(16) operations whose input length is already 16 are replaced by parameter-free Identity modules; prediction semantics and checkpoint parameter keys are unchanged.",
        ],
    }
    report_reference = store.publish_bytes(
        artifact_id=f"{execution_id}-report",
        relative_path=f"executions/{execution_id}/task-14.10-report.json",
        content=_json_bytes(report),
        media_type="application/json",
    )
    _register_artifact(registry, report_reference, "task_14_10_report", None)
    summary = {"status": "pass", "execution_id": execution_id, "report": report_reference.model_dump(mode="json"), "run_ids": [item["run_id"] for item in runs], "legacy_all_within_tolerance": all(item["comparison"]["all_within_tolerance"] for item in legacy_results.values())}
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the governed train-only baseline reproduction")
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        parser.error("--execute is required")
    try:
        result = execute_baseline()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
