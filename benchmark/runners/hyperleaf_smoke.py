"""Bounded, train-only HyperLeaf CUDA smoke for staged task 14.9.

The runner consumes a fast-lane launch manifest, reads only ``train.csv`` and
training cubes, and writes only below ``runs_v2/smoke/hyperleaf`` or
``artifacts``.  It deliberately has no official-test or solution-label path.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from benchmark.catalog.hyperleaf import (
    HYPERLEAF_TARGETS,
    GLOBAL_SPECTRUM_KEY,
    PCA_PSEUDO_RGB_KEY,
    VISIBLE_RGB_KEY,
    HyperLeaf2024Adapter,
    HyperLeafPartition,
    HyperLeafRepresentationBuilder,
)
from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import PredictionArtifactWriter
from benchmark.models.hyperleaf_route_a import (
    GLOBAL_FUSION_MODEL_KEY,
    SOURCE_REVISION,
    create_hyperleaf_route_a_registry,
)
from benchmark.recovery import FailureAtomicArtifactStore, RecoveryManager
from benchmark.registry import (
    ArtifactRecord,
    ClaimDecision,
    FrozenRunIdentity,
    Provenance,
    RunRecord,
    RunRegistry,
)
from benchmark.schema import (
    ClaimStatus,
    FitBoundary,
    FitBoundaryScope,
    FrozenRunConfig,
    PredictionKey,
    PredictionRecord,
    ResourceField,
    RunStatus,
    RuntimeMeasurements,
    assess_resource_budget,
    canonical_serialize,
    hash_file,
    hash_object,
)

WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "smoke.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "smoke" / "hyperleaf"
SHARED_RELATIVE_ROOT = "artifacts/smoke/hyperleaf"
FOLD_ID = "train-only-smoke"
PROTOCOL_KEY = "leaf.smoke.train-only.v1"
MAX_SAMPLES = 8
PREDICTION_SAMPLES = 2
BATCH_SIZE = 2


@dataclass(frozen=True)
class SmokePartition:
    fit_ids: tuple[str, ...]
    prediction_ids: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.fit_ids + self.prediction_ids


def partition_training_ids(
    sample_ids: Sequence[str],
    *,
    max_samples: int = MAX_SAMPLES,
    prediction_samples: int = PREDICTION_SAMPLES,
) -> SmokePartition:
    """Choose a deterministic tiny train-only fit/prediction split."""

    if max_samples < 4:
        raise ValueError("smoke max_samples must be at least four")
    if prediction_samples <= 0 or prediction_samples >= max_samples:
        raise ValueError("prediction_samples must be between one and max_samples - 1")
    canonical = tuple(sorted(set(sample_ids), key=lambda value: value.encode("utf-8")))
    if len(canonical) < max_samples:
        raise ValueError("not enough HyperLeaf training samples for the bounded smoke")
    selected = canonical[:max_samples]
    split_at = max_samples - prediction_samples
    return SmokePartition(selected[:split_at], selected[split_at:])


def make_prediction_records(
    sample_ids: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    run_seed: int = 42,
) -> tuple[PredictionRecord, ...]:
    """Build finite vector regression records for one train-only smoke split."""

    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    expected_shape = (len(sample_ids), len(HYPERLEAF_TARGETS))
    if truth.shape != expected_shape or prediction.shape != expected_shape:
        raise ValueError(f"prediction arrays must both have shape {expected_shape}")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("smoke predictions and labels must be finite")
    return tuple(
        PredictionRecord(
            key=PredictionKey(
                sample_id=sample_id,
                fold_id=FOLD_ID,
                seed=run_seed,
                inference_branch="global-fusion-smoke",
            ),
            target_name="three-target-output",
            y_true=tuple(float(value) for value in truth[index]),
            y_pred=tuple(float(value) for value in prediction[index]),
        )
        for index, sample_id in enumerate(sample_ids)
    )


def _json_bytes(value: Any) -> bytes:
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


def _torch_bytes(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _torch_load(payload: bytes, device: torch.device) -> Any:
    return torch.load(io.BytesIO(payload), map_location=device, weights_only=True)


def _tree_fingerprint(root: Path) -> str:
    if not root.exists():
        return hash_object({"exists": False})
    entries = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix().encode("utf-8")):
        stat = path.stat()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "is_file": path.is_file(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return hash_object(entries)


def _validate_g_drive(path: Path, name: str) -> None:
    drive = PureWindowsPath(str(path)).drive.casefold()
    if drive != "g:":
        raise RuntimeError(f"{name} must remain on the G: drive")


def _load_launch(path: Path) -> tuple[dict[str, Any], FrozenRunConfig, FrozenRunIdentity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("launch manifest must contain an object")
    required = {
        "operation": "run",
        "training_started": False,
        "dataset_key": "hyperleaf2024",
        "stage": "smoke",
        "model_key": GLOBAL_FUSION_MODEL_KEY,
        "protocol_key": PROTOCOL_KEY,
        "seed": 42,
        "device": "cuda:0",
        "process_count": 1,
        "numeric_precision": "float32",
        "determinism": "deterministic",
        "output_root": "runs_v2/smoke",
    }
    mismatches = [name for name, expected in required.items() if payload.get(name) != expected]
    if mismatches:
        raise ValueError("launch manifest differs from task 14.9 contract: " + ", ".join(mismatches))
    config = FrozenRunConfig.model_validate(payload["frozen_config"])
    identity = FrozenRunIdentity.model_validate(payload["frozen_identity"])
    if config.config_hash != identity.config_hash:
        raise ValueError("launch config and frozen identity hashes differ")
    if config.input_volume_bytes <= 0:
        raise ValueError("smoke input volume must be measured before launch")
    return payload, config, identity


def _identity_material(
    adapter: HyperLeaf2024Adapter,
    partition: SmokePartition,
) -> tuple[str, str, int, dict[str, str]]:
    targets_boundary = FitBoundary(
        fold_id=FOLD_ID,
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=partition.all_ids,
    )
    targets = adapter.training_targets(
        partition.all_ids,
        targets_boundary,
        require_complete_boundary=True,
    )
    cube_hashes: dict[str, str] = {}
    input_volume = 0
    for sample_id in partition.all_ids:
        cube_path = adapter.cube_path(sample_id)
        _validate_g_drive(cube_path, "HyperLeaf cube")
        cube_hashes[sample_id] = hash_file(cube_path)
        input_volume += cube_path.stat().st_size
    dataset_hash = hash_object(
        {
            "dataset_key": "hyperleaf2024",
            "partition": "train",
            "sample_ids": partition.all_ids,
            "cube_hashes": cube_hashes,
            "target_names": HYPERLEAF_TARGETS,
            "target_values": tuple(tuple(float(value) for value in row) for row in targets),
        }
    )
    protocol_hash = hash_object(
        {
            "protocol_key": PROTOCOL_KEY,
            "dataset_hash": dataset_hash,
            "fold_id": FOLD_ID,
            "fit_ids": partition.fit_ids,
            "prediction_ids": partition.prediction_ids,
            "official_evaluation_budget_consumed": 0,
        }
    )
    return dataset_hash, protocol_hash, input_volume, cube_hashes


def prepare_job_request(relative_path: str) -> Path:
    """Measure the tiny train-only identity and atomically write a launch request."""

    _validate_g_drive(WORKSPACE, "workspace")
    _validate_g_drive(DATA_ROOT, "HyperLeaf data root")
    normalized = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
    if not normalized.startswith("artifacts/"):
        raise ValueError("prepared job config must be written below artifacts/")
    adapter = HyperLeaf2024Adapter(DATA_ROOT, partition=HyperLeafPartition.TRAIN)
    partition = partition_training_ids(adapter.sample_ids(HyperLeafPartition.TRAIN))
    dataset_hash, protocol_hash, input_volume, _ = _identity_material(adapter, partition)
    request = {
        "schema_version": "1.0",
        "stage": "smoke",
        "job_name": "hyperleaf-smoke-task-14-9",
        "dataset_key": "hyperleaf2024",
        "dataset_hash": dataset_hash,
        "protocol_key": PROTOCOL_KEY,
        "protocol_hash": protocol_hash,
        "model_key": GLOBAL_FUSION_MODEL_KEY,
        "fold_id": FOLD_ID,
        "seed": 42,
        "input_volume_bytes": input_volume,
        "selection_metric_direction": "maximize",
        "parameters": {
            "batch_size": BATCH_SIZE,
            "dataset_split": "train-only",
            "max_samples": MAX_SAMPLES,
            "prediction_samples": PREDICTION_SAMPLES,
            "official_test_labels_allowed": False,
            "representation_keys": [VISIBLE_RGB_KEY, GLOBAL_SPECTRUM_KEY, PCA_PSEUDO_RGB_KEY],
        },
    }
    store = FailureAtomicArtifactStore(WORKSPACE)
    store.publish_bytes(
        artifact_id="hyperleaf-smoke-task-14-9-job-request",
        relative_path=normalized,
        content=_json_bytes(request),
        media_type="application/json",
    )
    return WORKSPACE / Path(*PurePosixPath(normalized).parts)


def _allocate_run(
    registry: RunRegistry,
    identity: FrozenRunIdentity,
    config: FrozenRunConfig,
    source_hash: str,
) -> RunRecord:
    registration = create_hyperleaf_route_a_registry().resolve(GLOBAL_FUSION_MODEL_KEY)
    return registry.allocate_run(
        frozen_identity=identity,
        model_identity=registration.identity,
        input_contract=registration.input_contract,
        provenance=Provenance(
            code_revision=SOURCE_REVISION,
            source_hash=source_hash,
            environment_hash=identity.environment_lock_hash,
            config_hash=identity.config_hash,
            dataset_hash=identity.dataset_hash,
            adaptation_notes=("task-14.9-train-only-bounded-smoke",),
        ),
        resource_budget=config.resource_budget,
        claim_decision=ClaimDecision(
            status=ClaimStatus.SANITY_ONLY,
            supporting_reasons=("bounded_train_only_cuda_smoke",),
            blocking_reasons=("not_formal_evaluation",),
            prerequisite_checks={"official_test_labels_read": False},
        ),
    )


def _append_running(registry: RunRegistry, planned: RunRecord) -> RunRecord:
    running = RunRecord.model_validate(
        {**planned.model_dump(mode="python"), "status": RunStatus.RUNNING}
    )
    registry.append_run(running)
    return running


def _make_optimizer(model: torch.nn.Module) -> tuple[Any, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return optimizer, scheduler


def _train_epoch(
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    rgb: Tensor,
    spectrum: Tensor,
    targets: Tensor,
) -> tuple[float, int, bool]:
    model.train()
    losses: list[float] = []
    gradients_finite = True
    steps = 0
    for start in range(0, len(rgb), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(rgb))
        prediction = model(rgb[start:end], spectrum[start:end])
        loss = torch.nn.functional.mse_loss(prediction, targets[start:end])
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        if not gradients_finite:
            raise RuntimeError("non-finite smoke gradient")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        steps += 1
    scheduler.step()
    return float(sum(losses) / len(losses)), steps, gradients_finite


def _state_bytes(
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    *,
    epoch: int,
    steps: int,
) -> dict[str, bytes]:
    return {
        "model": _torch_bytes(model.state_dict()),
        "optimizer": _torch_bytes(optimizer.state_dict()),
        "scheduler": _torch_bytes(scheduler.state_dict()),
        "stage": _json_bytes({"completed_epoch": epoch, "completed_training_steps": steps}),
    }


def _budget_limit(config: FrozenRunConfig, field: ResourceField) -> int:
    return next(limit.value for limit in config.resource_budget.limits if limit.field == field.value)


def execute_smoke(launch_manifest: Path) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _validate_g_drive(WORKSPACE, "workspace")
    _validate_g_drive(DATA_ROOT, "HyperLeaf data root")
    launch, config, identity = _load_launch(launch_manifest)
    template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.SMOKE:
        raise RuntimeError("task 14.9 requires the frozen smoke template")
    live_environment = validate_live_environment(template)
    if identity.environment_lock_hash != live_environment.lock_hash:
        raise RuntimeError("live Environment_Lock differs from frozen run identity")

    readonly_before = {name: _tree_fingerprint(WORKSPACE / name) for name in ("src", "runs")}
    started_ns = time.monotonic_ns()
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    device = torch.device("cuda:0")

    adapter = HyperLeaf2024Adapter(DATA_ROOT, partition=HyperLeafPartition.TRAIN)
    partition = partition_training_ids(adapter.sample_ids(HyperLeafPartition.TRAIN))
    dataset_hash, protocol_hash, input_volume, cube_hashes = _identity_material(adapter, partition)
    observed_identity = {
        "dataset_hash": dataset_hash,
        "protocol_hash": protocol_hash,
        "input_volume_bytes": input_volume,
    }
    expected_identity = {
        "dataset_hash": identity.dataset_hash,
        "protocol_hash": identity.protocol_hash,
        "input_volume_bytes": config.input_volume_bytes,
    }
    if observed_identity != expected_identity:
        raise RuntimeError("measured train-only identity differs from frozen launch request")

    fit_boundary = FitBoundary(
        fold_id=FOLD_ID,
        scope=FitBoundaryScope.INNER_TRAIN,
        inner_fold_id="smoke-fit",
        allowed_fit_sample_ids=partition.fit_ids,
        outer_test_sample_ids=partition.prediction_ids,
    )
    prediction_label_boundary = FitBoundary(
        fold_id=FOLD_ID,
        scope=FitBoundaryScope.OUTER_TRAIN,
        allowed_fit_sample_ids=partition.prediction_ids,
    )
    builder = HyperLeafRepresentationBuilder(adapter)
    target_artifact = builder.fit_target_normalization(
        partition.fit_ids,
        fit_boundary,
        dataset_hash=dataset_hash,
        protocol_hash=protocol_hash,
    )
    pca_artifact = builder.fit_pca_pseudo_rgb(
        partition.fit_ids,
        fit_boundary,
        dataset_hash=dataset_hash,
        protocol_hash=protocol_hash,
    )

    all_rgb = np.stack([builder.visible_rgb(sample_id) for sample_id in partition.all_ids])
    all_spectrum = np.stack([builder.global_spectrum(sample_id) for sample_id in partition.all_ids])
    pca_sample = builder.pca_pseudo_rgb(partition.prediction_ids[0], pca_artifact, fit_boundary)
    if all_rgb.shape != (MAX_SAMPLES, 3, 32, 64):
        raise RuntimeError(f"unexpected visible RGB shape: {all_rgb.shape}")
    if all_spectrum.shape != (MAX_SAMPLES, 128):
        raise RuntimeError(f"unexpected global spectrum shape: {all_spectrum.shape}")
    if pca_sample.shape != (3, 48, 352):
        raise RuntimeError(f"unexpected PCA pseudo-RGB shape: {pca_sample.shape}")
    if not all(np.isfinite(value).all() for value in (all_rgb, all_spectrum, pca_sample)):
        raise RuntimeError("non-finite HyperLeaf representation")

    fit_targets_raw = adapter.training_targets(
        partition.fit_ids,
        fit_boundary,
        require_complete_boundary=True,
    )
    prediction_targets_raw = adapter.training_targets(
        partition.prediction_ids,
        prediction_label_boundary,
        require_complete_boundary=True,
    )
    fit_targets = builder.preprocessing.apply(target_artifact, fit_targets_raw, fit_boundary).astype(np.float32)
    fit_count = len(partition.fit_ids)
    train_rgb = torch.from_numpy(all_rgb[:fit_count]).to(device=device, dtype=torch.float32)
    train_spectrum = torch.from_numpy(all_spectrum[:fit_count]).to(device=device, dtype=torch.float32)
    train_targets = torch.from_numpy(fit_targets).to(device=device, dtype=torch.float32)
    prediction_rgb = torch.from_numpy(all_rgb[fit_count:]).to(device=device, dtype=torch.float32)
    prediction_spectrum = torch.from_numpy(all_spectrum[fit_count:]).to(device=device, dtype=torch.float32)

    output_store = FailureAtomicArtifactStore(OUTPUT_ROOT)
    registry = RunRegistry(OUTPUT_ROOT / "registry.jsonl")
    recovery = RecoveryManager(registry, output_store)
    source_hash = hash_object(
        {
            "runner": hash_file(Path(__file__)),
            "model_adapter": hash_file(WORKSPACE / "benchmark" / "models" / "hyperleaf_route_a.py"),
            "legacy_model_source": hash_file(WORKSPACE / "src" / "hypercross_routeA.py"),
        }
    )

    source_planned = _allocate_run(registry, identity, config, source_hash)
    _append_running(registry, source_planned)
    model = create_hyperleaf_route_a_registry().build(GLOBAL_FUSION_MODEL_KEY).to(device=device, dtype=torch.float32)
    optimizer, scheduler = _make_optimizer(model)
    epoch1_loss, epoch1_steps, gradients_finite = _train_epoch(
        model, optimizer, scheduler, train_rgb, train_spectrum, train_targets
    )
    states1 = _state_bytes(model, optimizer, scheduler, epoch=1, steps=epoch1_steps)
    recovery1 = recovery.save_checkpoint_boundary(
        run_id=source_planned.run_id,
        recovery_id=f"hyperleaf-smoke-{source_planned.run_id}-boundary-1",
        relative_path=f"recovery/{source_planned.run_id}/boundary-1.json",
        model_state=states1["model"],
        optimizer_state=states1["optimizer"],
        scheduler_state=states1["scheduler"],
        stage_state=states1["stage"],
        last_completed_stage="smoke_epoch_1",
        completed_checkpoint_boundary=1,
    )
    recovery.record_interruption(
        source_planned.run_id,
        status=RunStatus.ABORTED,
        last_completed_stage="smoke_epoch_1",
    )

    resumed_model = create_hyperleaf_route_a_registry().build(GLOBAL_FUSION_MODEL_KEY).to(
        device=device, dtype=torch.float32
    )
    resumed_optimizer, resumed_scheduler = _make_optimizer(resumed_model)
    restored_stage: dict[str, Any] = {}

    def restore_stage(payload: bytes) -> None:
        restored_stage.update(json.loads(payload.decode("utf-8")))

    restored = recovery.resume(
        source_planned.run_id,
        identity,
        restore_model=lambda payload: resumed_model.load_state_dict(_torch_load(payload, device)),
        restore_optimizer=lambda payload: resumed_optimizer.load_state_dict(_torch_load(payload, device)),
        restore_scheduler=lambda payload: resumed_scheduler.load_state_dict(_torch_load(payload, device)),
        restore_stage=restore_stage,
    )
    if restored_stage != {"completed_epoch": 1, "completed_training_steps": epoch1_steps}:
        raise RuntimeError("restored stage state differs from checkpoint boundary")
    model_equal = all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), resumed_model.state_dict().values())
    )
    if not model_equal:
        raise RuntimeError("restored model state differs from checkpoint boundary")

    resumed_planned = _allocate_run(registry, identity, config, source_hash)
    resumed_running = _append_running(registry, resumed_planned)
    epoch2_loss, epoch2_steps, epoch2_gradients_finite = _train_epoch(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        train_rgb,
        train_spectrum,
        train_targets,
    )
    total_steps = epoch1_steps + epoch2_steps
    states2 = _state_bytes(resumed_model, resumed_optimizer, resumed_scheduler, epoch=2, steps=total_steps)
    recovery2 = recovery.save_checkpoint_boundary(
        run_id=resumed_planned.run_id,
        recovery_id=f"hyperleaf-smoke-{resumed_planned.run_id}-boundary-2",
        relative_path=f"recovery/{resumed_planned.run_id}/boundary-2.json",
        model_state=states2["model"],
        optimizer_state=states2["optimizer"],
        scheduler_state=states2["scheduler"],
        stage_state=states2["stage"],
        last_completed_stage="smoke_epoch_2",
        completed_checkpoint_boundary=2,
    )

    elapsed_ms = (time.monotonic_ns() - started_ns) // 1_000_000
    peak_memory = int(torch.cuda.max_memory_allocated(0))
    registration = create_hyperleaf_route_a_registry().resolve(GLOBAL_FUSION_MODEL_KEY)
    measurements = RuntimeMeasurements(
        completed_epochs=2,
        completed_training_steps=total_steps,
        wall_clock_duration_ms=int(elapsed_ms),
        peak_device_memory_bytes=peak_memory,
        device_identity=live_environment.device_identity,
        numeric_precision="float32",
        parameter_count=registration.identity.parameter_count,
        input_volume_bytes=input_volume,
    )
    assessment = assess_resource_budget(config.resource_budget, measurements, checkpoint_boundary=2)
    if assessment.hard_exceeded:
        raise RuntimeError("smoke exceeded a frozen hard limit at checkpoint boundary 2")
    if measurements.completed_epochs > _budget_limit(config, ResourceField.COMPLETED_EPOCHS):
        raise RuntimeError("smoke exceeded the frozen epoch hard limit")
    if measurements.completed_training_steps > _budget_limit(config, ResourceField.COMPLETED_TRAINING_STEPS):
        raise RuntimeError("smoke exceeded the frozen step hard limit")

    checkpoint_refs = []
    for role, payload, epoch in (
        ("selected", states1["model"] if epoch1_loss <= epoch2_loss else states2["model"], 1 if epoch1_loss <= epoch2_loss else 2),
        ("last", states2["model"], 2),
    ):
        reference = output_store.publish_bytes(
            artifact_id=f"hyperleaf-smoke-{resumed_planned.run_id}-{role}",
            relative_path=f"checkpoints/{resumed_planned.run_id}/{role}-epoch-{epoch}.pt",
            content=payload,
            media_type="application/vnd.pytorch.state-dict",
        )
        registry.append_artifact(
            ArtifactRecord(
                artifact=reference,
                artifact_kind=f"smoke_{role}_checkpoint",
                run_id=resumed_planned.run_id,
                created_utc=datetime.now(timezone.utc),
            )
        )
        checkpoint_refs.append(reference)

    reloaded_model = create_hyperleaf_route_a_registry().build(GLOBAL_FUSION_MODEL_KEY).to(
        device=device, dtype=torch.float32
    )
    reloaded_model.load_state_dict(_torch_load(states2["model"], device))
    resumed_model.eval()
    reloaded_model.eval()
    with torch.no_grad():
        normalized_prediction = resumed_model(prediction_rgb, prediction_spectrum)
        reloaded_normalized_prediction = reloaded_model(prediction_rgb, prediction_spectrum)
    checkpoint_prediction_equal = bool(torch.equal(normalized_prediction, reloaded_normalized_prediction))
    if not checkpoint_prediction_equal or not bool(torch.isfinite(normalized_prediction).all().item()):
        raise RuntimeError("reloaded checkpoint prediction verification failed")
    mean = np.asarray(target_artifact.state["mean"], dtype=np.float64)
    scale = np.asarray(target_artifact.state["scale"], dtype=np.float64)
    prediction_raw = normalized_prediction.detach().cpu().numpy().astype(np.float64) * scale + mean
    records = make_prediction_records(
        partition.prediction_ids,
        prediction_targets_raw,
        prediction_raw,
    )
    prediction_writer = PredictionArtifactWriter(output_store)
    published_prediction = prediction_writer.publish_shards(
        artifact_id=f"hyperleaf-smoke-{resumed_planned.run_id}-predictions",
        run_id=resumed_planned.run_id,
        expected_sample_ids=partition.prediction_ids,
        shards=(records,),
        relative_path=f"predictions/{resumed_planned.run_id}.canonical.json",
    )
    registry.append_artifact(
        ArtifactRecord(
            artifact=published_prediction.reference,
            artifact_kind="smoke_prediction",
            run_id=resumed_planned.run_id,
            created_utc=datetime.now(timezone.utc),
        )
    )
    prediction_path = OUTPUT_ROOT / Path(*PurePosixPath(published_prediction.reference.relative_path).parts)
    reloaded_prediction = prediction_writer.reload(prediction_path.read_bytes())

    shared_store = FailureAtomicArtifactStore(WORKSPACE)
    transform_refs = []
    job_name = str(launch["job_name"])
    for name, artifact in (("target-normalization", target_artifact), ("pca", pca_artifact)):
        relative_path = f"{SHARED_RELATIVE_ROOT}/{job_name}/{name}.canonical.json"
        reference = shared_store.publish_bytes(
            artifact_id=f"hyperleaf-smoke-{job_name}-{name}",
            relative_path=relative_path,
            content=canonical_serialize(artifact),
            media_type="application/json",
        )
        registry.append_artifact(
            ArtifactRecord(
                artifact=reference,
                artifact_kind=f"smoke_{name}",
                run_id=resumed_planned.run_id,
                created_utc=datetime.now(timezone.utc),
            )
        )
        transform_refs.append(reference)

    completed = RunRecord.model_validate(
        {
            **resumed_running.model_dump(mode="python"),
            "status": RunStatus.COMPLETE,
            "transform_artifact_refs": tuple(transform_refs),
            "checkpoint_ref": checkpoint_refs[1],
            "recovery_artifact_ref": recovery2.artifact,
            "prediction_artifact_ref": published_prediction.reference,
            "metrics": {
                "smoke_epoch_losses": [epoch1_loss, epoch2_loss],
                "finite_loss": True,
                "backward_pass": True,
            },
            "runtime_measurements": measurements,
            "completed_utc": datetime.now(timezone.utc),
        }
    )
    registry.append_run(completed)

    readonly_after = {name: _tree_fingerprint(WORKSPACE / name) for name in ("src", "runs")}
    readonly_unchanged = readonly_before == readonly_after
    if not readonly_unchanged:
        raise RuntimeError("read-only src/ or runs/ metadata changed during smoke")

    report = {
        "schema_version": "1.0.0",
        "task_id": "14.9",
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "workspace": str(WORKSPACE),
            "data_root": str(DATA_ROOT),
            "write_roots": ["runs_v2/smoke/hyperleaf", "artifacts"],
            "src_and_runs_read_only": readonly_unchanged,
            "official_test_images_read": False,
            "official_test_labels_read": False,
            "official_solution_labels_read": False,
            "formal_evaluation_budget_consumed": 0,
            "baseline_or_strict_or_official_started": False,
        },
        "environment": {
            **live_environment.model_dump(mode="json"),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "torch_cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "process_count": 1,
            "precision": "float32",
            "execution_mode": "deterministic",
            "framework_settings": config.framework_settings.model_dump(mode="json"),
        },
        "identity": {
            "source_run_id": source_planned.run_id,
            "resumed_run_id": resumed_planned.run_id,
            "dataset_hash": dataset_hash,
            "protocol_hash": protocol_hash,
            "config_hash": config.config_hash,
            "environment_lock_hash": identity.environment_lock_hash,
            "model_key": identity.model_key,
            "fold_id": identity.fold_id,
            "seed": identity.seed,
        },
        "data": {
            "partition": "train-only",
            "sample_ids": list(partition.all_ids),
            "fit_sample_ids": list(partition.fit_ids),
            "prediction_sample_ids": list(partition.prediction_ids),
            "cube_content_hashes": cube_hashes,
            "target_names": list(HYPERLEAF_TARGETS),
            "visible_rgb_shape": list(all_rgb.shape),
            "global_spectrum_shape": list(all_spectrum.shape),
            "pca_pseudo_rgb_shape": list(pca_sample.shape),
            "dtypes": {"visible_rgb": str(all_rgb.dtype), "global_spectrum": str(all_spectrum.dtype), "pca_pseudo_rgb": str(pca_sample.dtype)},
            "representations_finite": True,
            "target_normalization_fit_ids_exact": target_artifact.fit_sample_ids == partition.fit_ids,
            "pca_fit_ids_exact": pca_artifact.fit_sample_ids == partition.fit_ids,
        },
        "training": {
            "epoch_losses": [epoch1_loss, epoch2_loss],
            "finite_loss": all(math.isfinite(value) for value in (epoch1_loss, epoch2_loss)),
            "backward_pass_verified": True,
            "gradients_finite": gradients_finite and epoch2_gradients_finite,
            "completed_epochs": 2,
            "completed_training_steps": total_steps,
        },
        "runtime_measurements": measurements.model_dump(mode="json"),
        "resource_budget": config.resource_budget.model_dump(mode="json"),
        "budget_assessment": assessment.model_dump(mode="json"),
        "checkpoint_resume": {
            "source_boundary": 1,
            "source_recovery_artifact": recovery1.artifact.model_dump(mode="json"),
            "restored_model": model_equal,
            "restored_optimizer": True,
            "restored_scheduler": True,
            "restored_stage": restored_stage,
            "resumed_boundary": 2,
            "resumed_recovery_artifact": recovery2.artifact.model_dump(mode="json"),
            "selected_checkpoint": checkpoint_refs[0].model_dump(mode="json"),
            "last_checkpoint": checkpoint_refs[1].model_dump(mode="json"),
            "last_checkpoint_reloaded": True,
            "reloaded_prediction_bitwise_equal": checkpoint_prediction_equal,
        },
        "prediction_artifact": {
            "reference": published_prediction.reference.model_dump(mode="json"),
            "record_count": len(reloaded_prediction.records),
            "expected_sample_ids": list(reloaded_prediction.expected_sample_ids),
            "complete": reloaded_prediction.complete,
            "hash_and_schema_reload_verified": reloaded_prediction == published_prediction.artifact,
            "per_sample_records_verified": len(reloaded_prediction.records) == len(partition.prediction_ids),
        },
    }
    report_path = f"{SHARED_RELATIVE_ROOT}/{job_name}/task-14.9-report.json"
    report_reference = shared_store.publish_bytes(
        artifact_id=f"hyperleaf-smoke-{job_name}-report",
        relative_path=report_path,
        content=_json_bytes(report),
        media_type="application/json",
    )
    report["report_reference"] = report_reference.model_dump(mode="json")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch-manifest", type=Path)
    group.add_argument("--prepare-job-config", type=str)
    arguments = parser.parse_args(argv)
    if arguments.prepare_job_config:
        path = prepare_job_request(arguments.prepare_job_config)
        print(json.dumps({"job_config": str(path)}, sort_keys=True))
        return 0
    try:
        report = execute_smoke(arguments.launch_manifest.resolve())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
