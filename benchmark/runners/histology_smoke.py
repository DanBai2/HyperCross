"""Bounded patient-disjoint Histology CUDA smoke for staged task 15.10.

The runner uses six immutable cached ROIs from three patients, validates one raw
cube through the governed read-only memmap API, and writes only below
``runs_v2/smoke/histology`` or ``artifacts``. It has no formal outer-evaluation
entry point and never consumes an evaluation budget.
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
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from benchmark.adapters.histology import DEFAULT_HISTOLOGY_ROOT, HistologyHSIGBAdapter
from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import PredictionArtifactWriter
from benchmark.models.histology import (
    LOCAL_ROUTE_A_INFERENCE_BRANCH,
    LOCAL_ROUTE_A_MODEL_KEY,
    SOURCE_REVISION,
    create_histology_model_registry,
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
    FrozenRunConfig,
    PredictionKey,
    PredictionRecord,
    ResourceField,
    RunStatus,
    RuntimeMeasurements,
    assess_resource_budget,
    hash_file,
    hash_object,
)

WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = DEFAULT_HISTOLOGY_ROOT
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "smoke.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "smoke" / "histology"
SHARED_RELATIVE_ROOT = "artifacts/smoke/histology"
GLOBAL_CACHE = "cache_routeA_v2"
LOCAL_CACHE = "cache_routeA_hist_mil"
FOLD_ID = "patient-disjoint-smoke"
PROTOCOL_KEY = "hist.smoke.patient-disjoint.v1"
FIT_PATIENT_COUNT = 2
PREDICTION_PATIENT_COUNT = 1
ROIS_PER_PATIENT = 2
BATCH_SIZE = 2


@dataclass(frozen=True)
class SmokePartition:
    fit_ids: tuple[str, ...]
    prediction_ids: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.fit_ids + self.prediction_ids

    @property
    def fit_groups(self) -> tuple[str, ...]:
        return tuple(sorted({sample.split("/", 1)[0] for sample in self.fit_ids}))

    @property
    def prediction_groups(self) -> tuple[str, ...]:
        return tuple(sorted({sample.split("/", 1)[0] for sample in self.prediction_ids}))


def select_patient_partition(
    samples_by_patient: dict[str, Sequence[str]],
    *,
    fit_patient_count: int = FIT_PATIENT_COUNT,
    prediction_patient_count: int = PREDICTION_PATIENT_COUNT,
    rois_per_patient: int = ROIS_PER_PATIENT,
) -> SmokePartition:
    """Select deterministic, class-representative ROIs from disjoint patients."""

    if fit_patient_count <= 0 or prediction_patient_count <= 0 or rois_per_patient < 2:
        raise ValueError("patient counts must be positive and rois_per_patient at least two")
    ordered_patients = sorted(samples_by_patient, key=_patient_sort_key)
    required = fit_patient_count + prediction_patient_count
    if len(ordered_patients) < required:
        raise ValueError("not enough patients for patient-disjoint smoke")

    def choose(patient: str) -> tuple[str, ...]:
        ids = tuple(sorted(set(samples_by_patient[patient]), key=lambda value: value.encode("utf-8")))
        tumor = [value for value in ids if value.endswith("_T") and not value.endswith("_NT")]
        non_tumor = [value for value in ids if value.endswith("_NT")]
        chosen = tumor[:1] + non_tumor[:1]
        remainder = [value for value in ids if value not in chosen]
        chosen.extend(remainder[: max(0, rois_per_patient - len(chosen))])
        if len(chosen) != rois_per_patient:
            raise ValueError(f"patient {patient} lacks enough smoke ROIs")
        return tuple(chosen)

    fit_patients = ordered_patients[:fit_patient_count]
    prediction_patients = ordered_patients[fit_patient_count:required]
    fit_ids = tuple(sample for patient in fit_patients for sample in choose(patient))
    prediction_ids = tuple(sample for patient in prediction_patients for sample in choose(patient))
    partition = SmokePartition(fit_ids=fit_ids, prediction_ids=prediction_ids)
    if set(partition.fit_groups) & set(partition.prediction_groups):
        raise RuntimeError("smoke patient groups overlap")
    return partition


def make_prediction_records(
    sample_ids: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    run_seed: int = 42,
) -> tuple[PredictionRecord, ...]:
    """Build one finite binary prediction record per patient-held-out ROI."""

    if len(sample_ids) != len(labels) or len(sample_ids) != len(probabilities):
        raise ValueError("sample_ids, labels, and probabilities must have equal length")
    records = []
    for sample_id, label, probability in zip(sample_ids, labels, probabilities):
        if label not in (0, 1):
            raise ValueError("Histology smoke labels must be binary")
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Histology smoke probabilities must be finite and in [0, 1]")
        records.append(
            PredictionRecord(
                key=PredictionKey(
                    sample_id=sample_id,
                    fold_id=FOLD_ID,
                    seed=run_seed,
                    inference_branch=LOCAL_ROUTE_A_INFERENCE_BRANCH,
                ),
                target_name="tumor",
                y_true=int(label),
                y_pred=int(value >= 0.5),
                probability=value,
            )
        )
    return tuple(records)


def _patient_sort_key(patient: str) -> tuple[int, str]:
    suffix = patient[1:] if patient.startswith("P") else ""
    return (int(suffix), patient) if suffix.isdigit() else (sys.maxsize, patient)


def _discover_partition(root: Path = DATA_ROOT) -> SmokePartition:
    samples: dict[str, tuple[str, ...]] = {}
    for patient_dir in root.iterdir():
        if patient_dir.is_dir() and patient_dir.name.startswith("P") and patient_dir.name[1:].isdigit():
            samples[patient_dir.name] = tuple(
                f"{patient_dir.name}/{roi.name}"
                for roi in patient_dir.iterdir()
                if roi.is_dir() and roi.name.startswith("ROI_")
            )
    return select_patient_partition(samples)


def _sample_label(sample_id: str) -> int:
    if sample_id.endswith("_NT"):
        return 0
    if sample_id.endswith("_T"):
        return 1
    raise ValueError(f"unrecognized Histology label suffix: {sample_id}")


def _cache_path(sample_id: str, cache_name: str) -> Path:
    patient, roi = sample_id.split("/", 1)
    return DATA_ROOT / cache_name / f"{patient}_{roi}.npz"


def _source_paths(sample_id: str) -> tuple[Path, ...]:
    patient, roi = sample_id.split("/", 1)
    roi_root = DATA_ROOT / patient / roi
    return (
        roi_root / "rgb.png",
        roi_root / "raw.hdr",
        _cache_path(sample_id, GLOBAL_CACHE),
        _cache_path(sample_id, LOCAL_CACHE),
    )


def _identity_material(partition: SmokePartition) -> tuple[str, str, int, dict[str, str]]:
    hashes: dict[str, str] = {}
    input_volume = 0
    for sample_id in partition.all_ids:
        for path in _source_paths(sample_id):
            _validate_g_drive(path, "Histology smoke input")
            if not path.is_file():
                raise FileNotFoundError(path)
            reference = path.relative_to(DATA_ROOT).as_posix()
            hashes[reference] = hash_file(path)
            input_volume += path.stat().st_size
    # One two-byte raw value is accessed to prove the governed memmap path.
    input_volume += np.dtype(np.uint16).itemsize
    dataset_hash = hash_object(
        {
            "dataset_key": "histology-hsi-gb",
            "sample_ids": partition.all_ids,
            "group_ids": {sample: sample.split("/", 1)[0] for sample in partition.all_ids},
            "labels": {sample: _sample_label(sample) for sample in partition.all_ids},
            "source_hashes": hashes,
        }
    )
    protocol_hash = hash_object(
        {
            "protocol_key": PROTOCOL_KEY,
            "dataset_hash": dataset_hash,
            "fold_id": FOLD_ID,
            "fit_ids": partition.fit_ids,
            "prediction_ids": partition.prediction_ids,
            "fit_groups": partition.fit_groups,
            "prediction_groups": partition.prediction_groups,
            "formal_evaluation_budget_consumed": 0,
        }
    )
    return dataset_hash, protocol_hash, input_volume, hashes


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


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
        entries.append({"path": path.relative_to(root).as_posix(), "is_file": path.is_file(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return hash_object(entries)


def _validate_g_drive(path: Path, name: str) -> None:
    if PureWindowsPath(str(path)).drive.casefold() != "g:":
        raise RuntimeError(f"{name} must remain on the G: drive")


def prepare_job_request(relative_path: str) -> Path:
    """Measure the tiny patient-disjoint identity and publish a job request."""

    _validate_g_drive(WORKSPACE, "workspace")
    _validate_g_drive(DATA_ROOT, "Histology data root")
    normalized = PurePosixPath(relative_path.replace("\\", "/")).as_posix()
    if not normalized.startswith("artifacts/"):
        raise ValueError("prepared job config must be written below artifacts/")
    partition = _discover_partition()
    dataset_hash, protocol_hash, input_volume, _ = _identity_material(partition)
    request = {
        "schema_version": "1.0",
        "stage": "smoke",
        "job_name": "histology-smoke-task-15-10",
        "dataset_key": "histology-hsi-gb",
        "dataset_hash": dataset_hash,
        "protocol_key": PROTOCOL_KEY,
        "protocol_hash": protocol_hash,
        "model_key": LOCAL_ROUTE_A_MODEL_KEY,
        "fold_id": FOLD_ID,
        "seed": 42,
        "input_volume_bytes": input_volume,
        "selection_metric_direction": "maximize",
        "parameters": {
            "batch_size": BATCH_SIZE,
            "fit_patient_count": FIT_PATIENT_COUNT,
            "prediction_patient_count": PREDICTION_PATIENT_COUNT,
            "rois_per_patient": ROIS_PER_PATIENT,
            "patient_grouped": True,
            "formal_evaluation_allowed": False,
            "global_cache": GLOBAL_CACHE,
            "local_cache": LOCAL_CACHE,
        },
    }
    store = FailureAtomicArtifactStore(WORKSPACE)
    store.publish_bytes(
        artifact_id="histology-smoke-task-15-10-job-request",
        relative_path=normalized,
        content=_json_bytes(request),
        media_type="application/json",
    )
    return WORKSPACE / Path(*PurePosixPath(normalized).parts)


def _load_launch(path: Path) -> tuple[dict[str, Any], FrozenRunConfig, FrozenRunIdentity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "operation": "run",
        "training_started": False,
        "dataset_key": "histology-hsi-gb",
        "stage": "smoke",
        "model_key": LOCAL_ROUTE_A_MODEL_KEY,
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
        raise ValueError("launch manifest differs from task 15.10 contract: " + ", ".join(mismatches))
    config = FrozenRunConfig.model_validate(payload["frozen_config"])
    identity = FrozenRunIdentity.model_validate(payload["frozen_identity"])
    if config.config_hash != identity.config_hash or config.input_volume_bytes <= 0:
        raise ValueError("launch identity/config mismatch or unmeasured input volume")
    return payload, config, identity


def _allocate_run(registry: RunRegistry, identity: FrozenRunIdentity, config: FrozenRunConfig, source_hash: str) -> RunRecord:
    registration = create_histology_model_registry().resolve(LOCAL_ROUTE_A_MODEL_KEY)
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
            adaptation_notes=("task-15.10-patient-disjoint-bounded-smoke",),
        ),
        resource_budget=config.resource_budget,
        claim_decision=ClaimDecision(
            status=ClaimStatus.SANITY_ONLY,
            supporting_reasons=("bounded_patient_disjoint_cuda_smoke",),
            blocking_reasons=("not_formal_evaluation",),
            prerequisite_checks={"formal_outer_labels_read": False},
        ),
    )


def _append_running(registry: RunRegistry, planned: RunRecord) -> RunRecord:
    running = RunRecord.model_validate({**planned.model_dump(mode="python"), "status": RunStatus.RUNNING})
    registry.append_run(running)
    return running


def _load_representations(partition: SmokePartition) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgbs, pcas, locals_, labels = [], [], [], []
    for sample_id in partition.all_ids:
        global_path = _cache_path(sample_id, GLOBAL_CACHE)
        local_path = _cache_path(sample_id, LOCAL_CACHE)
        with np.load(global_path, allow_pickle=False) as cached_global, np.load(local_path, allow_pickle=False) as cached_local:
            rgb = np.asarray(cached_global["v"], dtype=np.float32)
            pca = np.asarray(cached_global["s"], dtype=np.float32)
            local_rgb = np.asarray(cached_local["v"], dtype=np.float32)
            local_tokens = np.asarray(cached_local["s"], dtype=np.float32)
        # Independently decode RGB source to verify the declared source image shape.
        patient, roi = sample_id.split("/", 1)
        with Image.open(DATA_ROOT / patient / roi / "rgb.png") as image:
            decoded = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255.0
        if rgb.shape != (3, 64, 64) or decoded.shape != (3, 64, 64):
            raise RuntimeError(f"unexpected RGB shape for {sample_id}")
        if pca.shape != (128,) or local_tokens.shape != (16, 128):
            raise RuntimeError(f"unexpected PCA/local-token shape for {sample_id}")
        if local_rgb.shape != rgb.shape or not np.array_equal(local_rgb, rgb):
            raise RuntimeError(f"global/local cache RGB mismatch for {sample_id}")
        if not all(np.isfinite(value).all() for value in (rgb, pca, local_tokens, decoded)):
            raise RuntimeError(f"non-finite representation for {sample_id}")
        rgbs.append(rgb); pcas.append(pca); locals_.append(local_tokens); labels.append(_sample_label(sample_id))
    return np.stack(rgbs), np.stack(pcas), np.stack(locals_), np.asarray(labels, dtype=np.int64)


def _make_optimizer(model: torch.nn.Module) -> tuple[Any, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return optimizer, scheduler


def _build_smoke_model(device: torch.device) -> torch.nn.Module:
    """Build Local Route A with a deterministic equivalent 4x4 pooling op.

    CUDA does not provide a deterministic backward kernel for adaptive average
    pooling. With the fixed 64x64 smoke input, a 4x4 stride-4 average pool is
    mathematically equivalent to the model's adaptive 4x4 pool and has a
    deterministic CUDA backward implementation.
    """

    model = create_histology_model_registry().build(LOCAL_ROUTE_A_MODEL_KEY)
    adaptive = model.model.v[-1]
    if not isinstance(adaptive, torch.nn.AdaptiveAvgPool2d) or adaptive.output_size != (4, 4):
        raise RuntimeError("Local Route A pooling structure differs from smoke contract")
    model.model.v[-1] = torch.nn.AvgPool2d(kernel_size=4, stride=4)
    return model.to(device=device, dtype=torch.float32)


def _train_epoch(model: torch.nn.Module, optimizer: Any, scheduler: Any, rgb: Tensor, local_tokens: Tensor, labels: Tensor) -> tuple[float, int, bool]:
    model.train()
    losses: list[float] = []
    gradients_finite = True
    for start in range(0, len(rgb), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(rgb))
        logits = model(rgb[start:end], local_tokens[start:end])
        loss = torch.nn.functional.cross_entropy(logits, labels[start:end])
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients_finite = gradients_finite and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in model.parameters())
        if not gradients_finite:
            raise RuntimeError("non-finite smoke gradient")
        optimizer.step()
        losses.append(float(loss.detach().item()))
    scheduler.step()
    return float(sum(losses) / len(losses)), len(losses), gradients_finite


def _state_bytes(model: torch.nn.Module, optimizer: Any, scheduler: Any, *, epoch: int, steps: int) -> dict[str, bytes]:
    return {
        "model": _torch_bytes(model.state_dict()),
        "optimizer": _torch_bytes(optimizer.state_dict()),
        "scheduler": _torch_bytes(scheduler.state_dict()),
        "stage": _json_bytes({"completed_epoch": epoch, "completed_training_steps": steps}),
    }


def _budget_limit(config: FrozenRunConfig, field: ResourceField) -> int:
    return next(limit.value for limit in config.resource_budget.limits if limit.field == field.value)


def execute_smoke(launch_manifest: Path) -> dict[str, Any]:
    """Execute the two-epoch bounded smoke, including an intentional resume."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _validate_g_drive(WORKSPACE, "workspace")
    _validate_g_drive(DATA_ROOT, "Histology data root")
    launch, config, identity = _load_launch(launch_manifest)
    template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.SMOKE:
        raise RuntimeError("task 15.10 requires the frozen smoke template")
    live_environment = validate_live_environment(template)
    if identity.environment_lock_hash != live_environment.lock_hash:
        raise RuntimeError("live Environment_Lock differs from frozen run identity")

    readonly_before = {name: _tree_fingerprint(WORKSPACE / name) for name in ("src", "runs")}
    started_ns = time.monotonic_ns()
    torch.manual_seed(identity.seed)
    np.random.seed(identity.seed)
    torch.use_deterministic_algorithms(config.framework_settings.deterministic_algorithms)
    torch.backends.cudnn.deterministic = config.framework_settings.cudnn_deterministic
    torch.backends.cudnn.benchmark = config.framework_settings.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = config.framework_settings.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.framework_settings.allow_tf32
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    device = torch.device("cuda:0")

    partition = _discover_partition()
    dataset_hash, protocol_hash, input_volume, source_hashes = _identity_material(partition)
    if {"dataset_hash": dataset_hash, "protocol_hash": protocol_hash, "input_volume_bytes": input_volume} != {"dataset_hash": identity.dataset_hash, "protocol_hash": identity.protocol_hash, "input_volume_bytes": config.input_volume_bytes}:
        raise RuntimeError("measured patient-disjoint identity differs from frozen launch request")

    rgb_array, pca_array, local_array, labels_array = _load_representations(partition)
    adapter = HistologyHSIGBAdapter(DATA_ROOT)
    with adapter.open_raw_cube(partition.all_ids[0]) as cube:
        raw_shape = cube.shape
        raw_dtype = str(cube.dtype)
        raw_memmap_read_only = not cube.memmap_view().flags.writeable
        raw_probe = cube.read_window(bands=slice(0, 1), rows=slice(0, 1), columns=slice(0, 1))
        raw_probe_finite = bool(np.isfinite(raw_probe).all())
        raw_source_format = cube.source_format
    if not raw_memmap_read_only or raw_probe.shape != (1, 1, 1) or not raw_probe_finite:
        raise RuntimeError("governed raw-cube memmap verification failed")

    fit_count = len(partition.fit_ids)
    train_rgb = torch.from_numpy(rgb_array[:fit_count]).to(device=device, dtype=torch.float32)
    train_local = torch.from_numpy(local_array[:fit_count]).to(device=device, dtype=torch.float32)
    train_labels = torch.from_numpy(labels_array[:fit_count]).to(device=device, dtype=torch.long)
    prediction_rgb = torch.from_numpy(rgb_array[fit_count:]).to(device=device, dtype=torch.float32)
    prediction_local = torch.from_numpy(local_array[fit_count:]).to(device=device, dtype=torch.float32)

    output_store = FailureAtomicArtifactStore(OUTPUT_ROOT)
    registry = RunRegistry(OUTPUT_ROOT / "registry.jsonl")
    recovery = RecoveryManager(registry, output_store)
    source_hash = hash_object({
        "runner": hash_file(Path(__file__)),
        "model_adapter": hash_file(WORKSPACE / "benchmark" / "models" / "histology.py"),
        "legacy_model_source": hash_file(WORKSPACE / "src" / "hypercross_routeA.py"),
    })

    source_planned = _allocate_run(registry, identity, config, source_hash)
    _append_running(registry, source_planned)
    model = _build_smoke_model(device)
    optimizer, scheduler = _make_optimizer(model)
    epoch1_loss, epoch1_steps, gradients1 = _train_epoch(model, optimizer, scheduler, train_rgb, train_local, train_labels)
    states1 = _state_bytes(model, optimizer, scheduler, epoch=1, steps=epoch1_steps)
    recovery1 = recovery.save_checkpoint_boundary(
        run_id=source_planned.run_id,
        recovery_id=f"histology-smoke-{source_planned.run_id}-boundary-1",
        relative_path=f"recovery/{source_planned.run_id}/boundary-1.json",
        model_state=states1["model"], optimizer_state=states1["optimizer"], scheduler_state=states1["scheduler"], stage_state=states1["stage"],
        last_completed_stage="smoke_epoch_1", completed_checkpoint_boundary=1,
    )
    recovery.record_interruption(source_planned.run_id, status=RunStatus.ABORTED, last_completed_stage="smoke_epoch_1")

    resumed_model = _build_smoke_model(device)
    resumed_optimizer, resumed_scheduler = _make_optimizer(resumed_model)
    restored_stage: dict[str, Any] = {}
    restored = recovery.resume(
        source_planned.run_id, identity,
        restore_model=lambda payload: resumed_model.load_state_dict(_torch_load(payload, device)),
        restore_optimizer=lambda payload: resumed_optimizer.load_state_dict(_torch_load(payload, device)),
        restore_scheduler=lambda payload: resumed_scheduler.load_state_dict(_torch_load(payload, device)),
        restore_stage=lambda payload: restored_stage.update(json.loads(payload.decode("utf-8"))),
    )
    model_equal = all(torch.equal(left, right) for left, right in zip(model.state_dict().values(), resumed_model.state_dict().values()))
    if restored_stage != {"completed_epoch": 1, "completed_training_steps": epoch1_steps} or not model_equal:
        raise RuntimeError("recovery state differs from checkpoint boundary")

    resumed_planned = _allocate_run(registry, identity, config, source_hash)
    resumed_running = _append_running(registry, resumed_planned)
    epoch2_loss, epoch2_steps, gradients2 = _train_epoch(resumed_model, resumed_optimizer, resumed_scheduler, train_rgb, train_local, train_labels)
    total_steps = epoch1_steps + epoch2_steps
    states2 = _state_bytes(resumed_model, resumed_optimizer, resumed_scheduler, epoch=2, steps=total_steps)
    recovery2 = recovery.save_checkpoint_boundary(
        run_id=resumed_planned.run_id,
        recovery_id=f"histology-smoke-{resumed_planned.run_id}-boundary-2",
        relative_path=f"recovery/{resumed_planned.run_id}/boundary-2.json",
        model_state=states2["model"], optimizer_state=states2["optimizer"], scheduler_state=states2["scheduler"], stage_state=states2["stage"],
        last_completed_stage="smoke_epoch_2", completed_checkpoint_boundary=2,
    )

    elapsed_ms = (time.monotonic_ns() - started_ns) // 1_000_000
    registration = create_histology_model_registry().resolve(LOCAL_ROUTE_A_MODEL_KEY)
    measurements = RuntimeMeasurements(
        completed_epochs=2,
        completed_training_steps=total_steps,
        wall_clock_duration_ms=int(elapsed_ms),
        peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
        device_identity=live_environment.device_identity,
        numeric_precision="float32",
        parameter_count=registration.identity.parameter_count,
        input_volume_bytes=input_volume,
    )
    assessment = assess_resource_budget(config.resource_budget, measurements, checkpoint_boundary=2)
    if assessment.hard_exceeded:
        raise RuntimeError("smoke exceeded a frozen hard limit at checkpoint boundary 2")
    if measurements.completed_epochs > _budget_limit(config, ResourceField.COMPLETED_EPOCHS) or measurements.completed_training_steps > _budget_limit(config, ResourceField.COMPLETED_TRAINING_STEPS):
        raise RuntimeError("smoke exceeded a frozen epoch or step hard limit")

    checkpoint_refs = []
    selected_epoch = 1 if epoch1_loss <= epoch2_loss else 2
    selected_state = states1["model"] if selected_epoch == 1 else states2["model"]
    for role, payload, epoch in (("selected", selected_state, selected_epoch), ("last", states2["model"], 2)):
        reference = output_store.publish_bytes(
            artifact_id=f"histology-smoke-{resumed_planned.run_id}-{role}",
            relative_path=f"checkpoints/{resumed_planned.run_id}/{role}-epoch-{epoch}.pt",
            content=payload, media_type="application/vnd.pytorch.state-dict",
        )
        registry.append_artifact(ArtifactRecord(artifact=reference, artifact_kind=f"smoke_{role}_checkpoint", run_id=resumed_planned.run_id, created_utc=datetime.now(timezone.utc)))
        checkpoint_refs.append(reference)

    reloaded_model = _build_smoke_model(device)
    reloaded_model.load_state_dict(_torch_load(states2["model"], device))
    resumed_model.eval(); reloaded_model.eval()
    with torch.no_grad():
        logits = resumed_model(prediction_rgb, prediction_local)
        reloaded_logits = reloaded_model(prediction_rgb, prediction_local)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    prediction_equal = bool(torch.equal(logits, reloaded_logits))
    if not prediction_equal or not bool(torch.isfinite(probabilities).all().item()):
        raise RuntimeError("reloaded checkpoint prediction verification failed")
    records = make_prediction_records(partition.prediction_ids, labels_array[fit_count:].tolist(), probabilities.detach().cpu().tolist())
    prediction_writer = PredictionArtifactWriter(output_store)
    published_prediction = prediction_writer.publish_shards(
        artifact_id=f"histology-smoke-{resumed_planned.run_id}-predictions",
        run_id=resumed_planned.run_id,
        expected_sample_ids=partition.prediction_ids,
        shards=(records,),
        relative_path=f"predictions/{resumed_planned.run_id}.canonical.json",
    )
    registry.append_artifact(ArtifactRecord(artifact=published_prediction.reference, artifact_kind="smoke_prediction", run_id=resumed_planned.run_id, created_utc=datetime.now(timezone.utc)))
    prediction_path = OUTPUT_ROOT / Path(*PurePosixPath(published_prediction.reference.relative_path).parts)
    reloaded_prediction = prediction_writer.reload(prediction_path.read_bytes())

    completed = RunRecord.model_validate({
        **resumed_running.model_dump(mode="python"),
        "status": RunStatus.COMPLETE,
        "checkpoint_ref": checkpoint_refs[1],
        "recovery_artifact_ref": recovery2.artifact,
        "prediction_artifact_ref": published_prediction.reference,
        "metrics": {"smoke_epoch_losses": [epoch1_loss, epoch2_loss], "finite_loss": True, "backward_pass": True},
        "runtime_measurements": measurements,
        "completed_utc": datetime.now(timezone.utc),
    })
    registry.append_run(completed)

    readonly_after = {name: _tree_fingerprint(WORKSPACE / name) for name in ("src", "runs")}
    if readonly_before != readonly_after:
        raise RuntimeError("read-only src/ or runs/ metadata changed during smoke")

    report = {
        "schema_version": "1.0.0", "task_id": "15.10", "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "workspace": str(WORKSPACE), "dataset_key": "histology-hsi-gb",
            "write_roots": ["runs_v2/smoke/histology", "artifacts"],
            "src_and_runs_read_only": True, "f_drive_references_followed": False,
            "formal_evaluation_budget_consumed": 0,
            "baseline_strict_lopo_nested_or_outer_evaluation_started": False,
        },
        "environment": {
            **live_environment.model_dump(mode="json"), "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "torch_cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0), "device_capability": list(torch.cuda.get_device_capability(0)),
            "visible_cuda_device_count": torch.cuda.device_count(), "process_count": 1, "precision": "float32",
            "execution_mode": "deterministic", "framework_settings": config.framework_settings.model_dump(mode="json"),
        },
        "identity": {
            "source_run_id": source_planned.run_id, "resumed_run_id": resumed_planned.run_id,
            "dataset_hash": dataset_hash, "protocol_hash": protocol_hash, "config_hash": config.config_hash,
            "environment_lock_hash": identity.environment_lock_hash, "model_key": identity.model_key,
            "fold_id": identity.fold_id, "seed": identity.seed,
        },
        "data": {
            "sample_ids": list(partition.all_ids), "fit_sample_ids": list(partition.fit_ids), "prediction_sample_ids": list(partition.prediction_ids),
            "fit_group_ids": list(partition.fit_groups), "prediction_group_ids": list(partition.prediction_groups),
            "group_sets_disjoint": set(partition.fit_groups).isdisjoint(partition.prediction_groups),
            "labels": labels_array.tolist(), "source_hashes": source_hashes,
            "rgb_shape": list(rgb_array.shape), "pca_shape": list(pca_array.shape), "local_token_shape": list(local_array.shape),
            "dtypes": {"rgb": str(rgb_array.dtype), "pca": str(pca_array.dtype), "local_tokens": str(local_array.dtype)},
            "representations_finite": True,
            "raw_cube": {"shape": list(raw_shape), "dtype": raw_dtype, "source_format": raw_source_format, "memmap_read_only": raw_memmap_read_only, "probe_shape": list(raw_probe.shape), "probe_finite": raw_probe_finite},
        },
        "training": {
            "epoch_losses": [epoch1_loss, epoch2_loss], "finite_loss": all(math.isfinite(value) for value in (epoch1_loss, epoch2_loss)),
            "backward_pass_verified": True, "gradients_finite": gradients1 and gradients2,
            "completed_epochs": 2, "completed_training_steps": total_steps,
        },
        "runtime_measurements": measurements.model_dump(mode="json"),
        "resource_budget": config.resource_budget.model_dump(mode="json"), "budget_assessment": assessment.model_dump(mode="json"),
        "checkpoint_resume": {
            "source_boundary": 1, "source_recovery_artifact": recovery1.artifact.model_dump(mode="json"),
            "restored_model": model_equal, "restored_optimizer": True, "restored_scheduler": True, "restored_stage": restored_stage,
            "restored_recovery_id": restored.recovery.recovery_id, "resumed_boundary": 2,
            "resumed_recovery_artifact": recovery2.artifact.model_dump(mode="json"),
            "selected_checkpoint": checkpoint_refs[0].model_dump(mode="json"), "last_checkpoint": checkpoint_refs[1].model_dump(mode="json"),
            "last_checkpoint_reloaded": True, "reloaded_prediction_bitwise_equal": prediction_equal,
        },
        "prediction_artifact": {
            "reference": published_prediction.reference.model_dump(mode="json"), "record_count": len(reloaded_prediction.records),
            "expected_sample_ids": list(reloaded_prediction.expected_sample_ids), "complete": reloaded_prediction.complete,
            "hash_and_schema_reload_verified": reloaded_prediction == published_prediction.artifact,
            "per_sample_records_verified": len(reloaded_prediction.records) == len(partition.prediction_ids),
        },
    }
    shared_store = FailureAtomicArtifactStore(WORKSPACE)
    report_path = f"{SHARED_RELATIVE_ROOT}/{launch['job_name']}/task-15.10-report.json"
    report_reference = shared_store.publish_bytes(
        artifact_id=f"histology-smoke-{launch['job_name']}-report",
        relative_path=report_path, content=_json_bytes(report), media_type="application/json",
    )
    report["report_reference"] = report_reference.model_dump(mode="json")
    return report


def finalize_completed_run(launch_manifest: Path, run_id: str) -> dict[str, Any]:
    """Publish the report for an already completed smoke without retraining."""

    launch, config, identity = _load_launch(launch_manifest)
    template = load_fast_lane_template(TEMPLATE_PATH)
    live_environment = validate_live_environment(template)
    registry = RunRegistry(OUTPUT_ROOT / "registry.jsonl")
    completed = registry.latest_run(run_id)
    if completed.status is not RunStatus.COMPLETE or completed.runtime_measurements is None:
        raise RuntimeError("finalization requires an existing complete smoke run")
    if completed.frozen_identity != identity:
        raise RuntimeError("completed run identity differs from the immutable launch")

    partition = _discover_partition()
    dataset_hash, protocol_hash, input_volume, source_hashes = _identity_material(partition)
    if (dataset_hash, protocol_hash, input_volume) != (identity.dataset_hash, identity.protocol_hash, config.input_volume_bytes):
        raise RuntimeError("current smoke inputs differ from the completed run identity")
    rgb_array, pca_array, local_array, labels_array = _load_representations(partition)
    adapter = HistologyHSIGBAdapter(DATA_ROOT)
    with adapter.open_raw_cube(partition.all_ids[0]) as cube:
        raw_shape, raw_dtype, raw_source_format = cube.shape, str(cube.dtype), cube.source_format
        raw_memmap_read_only = not cube.memmap_view().flags.writeable
        raw_probe = cube.read_window(bands=slice(0, 1), rows=slice(0, 1), columns=slice(0, 1))

    if completed.checkpoint_ref is None or completed.prediction_artifact_ref is None or completed.recovery_artifact_ref is None:
        raise RuntimeError("completed run lacks a required smoke artifact")
    checkpoint_path = OUTPUT_ROOT / Path(*PurePosixPath(completed.checkpoint_ref.relative_path).parts)
    prediction_path = OUTPUT_ROOT / Path(*PurePosixPath(completed.prediction_artifact_ref.relative_path).parts)
    recovery_path = OUTPUT_ROOT / Path(*PurePosixPath(completed.recovery_artifact_ref.relative_path).parts)
    for path, reference in ((checkpoint_path, completed.checkpoint_ref), (prediction_path, completed.prediction_artifact_ref), (recovery_path, completed.recovery_artifact_ref)):
        if hash_file(path) != reference.content_hash:
            raise RuntimeError(f"completed artifact hash mismatch: {path.name}")

    prediction_writer = PredictionArtifactWriter(FailureAtomicArtifactStore(OUTPUT_ROOT))
    prediction = prediction_writer.reload(prediction_path.read_bytes())
    if not prediction.complete or tuple(prediction.expected_sample_ids) != partition.prediction_ids or len(prediction.records) != len(partition.prediction_ids):
        raise RuntimeError("completed prediction artifact is incomplete")

    # Re-run inference only, with no optimizer or fitted-state updates, to verify
    # that the retained checkpoint reproduces the published probabilities.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(identity.seed)
    torch.use_deterministic_algorithms(config.framework_settings.deterministic_algorithms)
    torch.backends.cudnn.deterministic = config.framework_settings.cudnn_deterministic
    torch.backends.cudnn.benchmark = config.framework_settings.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = config.framework_settings.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.framework_settings.allow_tf32
    device = torch.device("cuda:0")
    model = _build_smoke_model(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    fit_count = len(partition.fit_ids)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(rgb_array[fit_count:]).to(device=device, dtype=torch.float32),
            torch.from_numpy(local_array[fit_count:]).to(device=device, dtype=torch.float32),
        )
        observed_probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
    recorded_probabilities = [float(record.probability) for record in prediction.records]
    if observed_probabilities != recorded_probabilities:
        raise RuntimeError("retained checkpoint does not reproduce prediction artifact")

    history = []
    with (OUTPUT_ROOT / "registry.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("entry_kind") == "run":
                record = entry["record"]
                if record.get("status") == "aborted" and record.get("frozen_identity") == identity.model_dump(mode="json"):
                    history.append(record)
    if not history:
        raise RuntimeError("completed run lacks the intentional interruption record")
    source_record = history[-1]
    source_run_id = source_record["run_id"]
    recovery1 = registry.latest_recovery(source_run_id)
    recovery2 = registry.latest_recovery(run_id)
    selected = registry.get_artifact(f"histology-smoke-{run_id}-selected").artifact
    assessment = assess_resource_budget(config.resource_budget, completed.runtime_measurements, checkpoint_boundary=2)
    if assessment.hard_exceeded:
        raise RuntimeError("completed smoke measurements exceed a frozen hard limit")

    report = {
        "schema_version": "1.0.0", "task_id": "15.10", "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "finalization_mode": "existing_completed_run_no_retraining",
        "scope": {
            "workspace": str(WORKSPACE), "dataset_key": "histology-hsi-gb",
            "write_roots": ["runs_v2/smoke/histology", "artifacts"],
            "src_and_runs_read_only": True, "f_drive_references_followed": False,
            "formal_evaluation_budget_consumed": 0,
            "baseline_strict_lopo_nested_or_outer_evaluation_started": False,
        },
        "environment": {
            **live_environment.model_dump(mode="json"), "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "torch_cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0), "device_capability": list(torch.cuda.get_device_capability(0)),
            "visible_cuda_device_count": torch.cuda.device_count(), "process_count": 1, "precision": "float32",
            "execution_mode": "deterministic", "framework_settings": config.framework_settings.model_dump(mode="json"),
        },
        "identity": {
            "source_run_id": source_run_id, "resumed_run_id": run_id,
            "dataset_hash": dataset_hash, "protocol_hash": protocol_hash, "config_hash": config.config_hash,
            "environment_lock_hash": identity.environment_lock_hash, "model_key": identity.model_key,
            "fold_id": identity.fold_id, "seed": identity.seed,
        },
        "data": {
            "sample_ids": list(partition.all_ids), "fit_sample_ids": list(partition.fit_ids), "prediction_sample_ids": list(partition.prediction_ids),
            "fit_group_ids": list(partition.fit_groups), "prediction_group_ids": list(partition.prediction_groups),
            "group_sets_disjoint": set(partition.fit_groups).isdisjoint(partition.prediction_groups),
            "labels": labels_array.tolist(), "source_hashes": source_hashes,
            "rgb_shape": list(rgb_array.shape), "pca_shape": list(pca_array.shape), "local_token_shape": list(local_array.shape),
            "dtypes": {"rgb": str(rgb_array.dtype), "pca": str(pca_array.dtype), "local_tokens": str(local_array.dtype)},
            "representations_finite": bool(np.isfinite(rgb_array).all() and np.isfinite(pca_array).all() and np.isfinite(local_array).all()),
            "raw_cube": {"shape": list(raw_shape), "dtype": raw_dtype, "source_format": raw_source_format, "memmap_read_only": raw_memmap_read_only, "probe_shape": list(raw_probe.shape), "probe_finite": bool(np.isfinite(raw_probe).all())},
        },
        "training": {
            "epoch_losses": completed.metrics["smoke_epoch_losses"], "finite_loss": completed.metrics["finite_loss"],
            "backward_pass_verified": completed.metrics["backward_pass"], "gradients_finite": True,
            "completed_epochs": completed.runtime_measurements.completed_epochs,
            "completed_training_steps": completed.runtime_measurements.completed_training_steps,
            "deterministic_pooling_glue": "AdaptiveAvgPool2d((4,4)) replaced by equivalent AvgPool2d(4,4) for fixed 64x64 smoke input",
        },
        "runtime_measurements": completed.runtime_measurements.model_dump(mode="json"),
        "resource_budget": config.resource_budget.model_dump(mode="json"), "budget_assessment": assessment.model_dump(mode="json"),
        "checkpoint_resume": {
            "source_boundary": 1, "source_recovery_artifact": recovery1.artifact.model_dump(mode="json"),
            "restored_model": True, "restored_optimizer": True, "restored_scheduler": True,
            "restored_stage": {"completed_epoch": 1, "completed_training_steps": 2},
            "restored_recovery_id": recovery1.recovery_id, "resumed_boundary": 2,
            "resumed_recovery_artifact": recovery2.artifact.model_dump(mode="json"),
            "selected_checkpoint": selected.model_dump(mode="json"), "last_checkpoint": completed.checkpoint_ref.model_dump(mode="json"),
            "last_checkpoint_reloaded": True, "reloaded_prediction_bitwise_equal": True,
        },
        "prediction_artifact": {
            "reference": completed.prediction_artifact_ref.model_dump(mode="json"), "record_count": len(prediction.records),
            "expected_sample_ids": list(prediction.expected_sample_ids), "complete": prediction.complete,
            "hash_and_schema_reload_verified": True, "per_sample_records_verified": True,
            "checkpoint_probabilities_reproduced_exactly": True,
        },
    }
    shared_store = FailureAtomicArtifactStore(WORKSPACE)
    report_reference = shared_store.publish_bytes(
        artifact_id=f"histology-smoke-{launch['job_name']}-report",
        relative_path=f"{SHARED_RELATIVE_ROOT}/{launch['job_name']}/task-15.10-report.json",
        content=_json_bytes(report), media_type="application/json",
    )
    report["report_reference"] = report_reference.model_dump(mode="json")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch-manifest", type=Path)
    group.add_argument("--prepare-job-config", type=str)
    group.add_argument("--finalize-run-id", type=str)
    parser.add_argument("--finalize-launch-manifest", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.prepare_job_config:
        path = prepare_job_request(arguments.prepare_job_config)
        print(json.dumps({"job_config": str(path)}, sort_keys=True))
        return 0
    try:
        if arguments.finalize_run_id:
            if arguments.finalize_launch_manifest is None:
                raise ValueError("--finalize-launch-manifest is required with --finalize-run-id")
            report = finalize_completed_run(
                arguments.finalize_launch_manifest.resolve(), arguments.finalize_run_id
            )
        else:
            report = execute_smoke(arguments.launch_manifest.resolve())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
