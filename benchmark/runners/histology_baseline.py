"""Governed reproduction of the Histology fixed-split development baselines.

Task 15.11 recreates the historical seed-42 patient-grouped 70/30 split. Legacy
checkpoints are loaded read-only for tolerance checks only because they do not
bind complete preprocessing lineage. New checkpoints, recovery state,
predictions, hashes, metrics, and caveats are written only below
``runs_v2/baselines/histology``. This runner never starts nested CV or LOPO.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import PredictionArtifactWriter, binary_metrics
from benchmark.models.checkpoint import ExternalCheckpointLoader
from benchmark.models.histology import (
    HistologyRandomForestAdapter,
    HistologySVMAdapter,
    LOCAL_ROUTE_A_INFERENCE_BRANCH,
    LOCAL_ROUTE_A_MODEL_KEY,
    MIL_CONCAT_INFERENCE_BRANCH,
    MIL_CONCAT_MODEL_KEY,
    RF_INFERENCE_BRANCH,
    RF_MODEL_KEY,
    SVM_INFERENCE_BRANCH,
    SVM_MODEL_KEY,
    create_histology_model_registry,
)
from benchmark.models.histology_baseline import (
    GLOBAL_ROUTE_A_INFERENCE_BRANCH,
    GLOBAL_ROUTE_A_MODEL_KEY,
    create_histology_baseline_registry,
)
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
    FrozenRunConfig,
    ImplementationKind,
    MetricSchema,
    NumericPrecision,
    PredictionKey,
    PredictionRecord,
    ResourceField,
    RunStatus,
    RuntimeMeasurements,
    TaskKind,
    assess_resource_budget,
    hash_file,
    hash_object,
)

WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\PKG - HistologyHSI-GB")
GLOBAL_CACHE = DATA_ROOT / "cache_routeA_v2"
LOCAL_CACHE = DATA_ROOT / "cache_routeA_hist_mil"
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "baseline.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "baselines" / "histology"
PROTOCOL_KEY = "hist.fixed9-4.seed42.legacy-reproduction.v1"
FOLD_ID = "legacy-patient70-30-seed42"
SEED = 42
BATCH_SIZE = 16
LEGACY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FixedSplit:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]


@dataclass(frozen=True)
class HistologyArrays:
    rgb: np.ndarray
    global_spectrum: np.ndarray
    local_tokens: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    group_by_sample: Mapping[str, str]
    cache_hashes: Mapping[str, Mapping[str, str]]
    input_volume_bytes: int


@dataclass(frozen=True)
class NeuralSpec:
    name: str
    model_key: str
    inference_branch: str
    epochs: int
    local_input: bool
    align_weight: float
    nce_weight: float
    balanced: bool
    legacy_checkpoint: str
    legacy_history: str
    legacy_checkpoint_hash: str


NEURAL_SPECS = (
    NeuralSpec(
        name="local_route_a",
        model_key=LOCAL_ROUTE_A_MODEL_KEY,
        inference_branch=LOCAL_ROUTE_A_INFERENCE_BRANCH,
        epochs=40,
        local_input=True,
        align_weight=0.08,
        nce_weight=0.04,
        balanced=False,
        legacy_checkpoint="runs/hist_20261005_230817/routeA_mil_local_best.pt",
        legacy_history="runs/hist_20261005_230817/routeA_mil_local_history.json",
        legacy_checkpoint_hash="fa44a4cf2768352216941e8f5fd12570b7ca1c19c3a1226c37d89a671c1c8937",
    ),
    NeuralSpec(
        name="mil_concat",
        model_key=MIL_CONCAT_MODEL_KEY,
        inference_branch=MIL_CONCAT_INFERENCE_BRANCH,
        epochs=40,
        local_input=True,
        align_weight=0.0,
        nce_weight=0.0,
        balanced=False,
        legacy_checkpoint="runs/hist_20261005_230529/mil_concat_best.pt",
        legacy_history="runs/hist_20261005_230529/mil_concat_history.json",
        legacy_checkpoint_hash="49333bd9e131184f39ea909708603c3ca95d3efdc898a28ba200c6bb6196cf04",
    ),
    NeuralSpec(
        name="global_route_a",
        model_key=GLOBAL_ROUTE_A_MODEL_KEY,
        inference_branch=GLOBAL_ROUTE_A_INFERENCE_BRANCH,
        epochs=40,
        local_input=False,
        align_weight=0.0,
        nce_weight=0.0,
        balanced=True,
        legacy_checkpoint="runs/hist_20261005_221608/routeA_best.pt",
        legacy_history="runs/hist_20261005_221608/routeA_history.json",
        legacy_checkpoint_hash="97556560e77bea894036b187dd6843358d83ff7d25c3b162c9140ba183dd3187",
    ),
)


def fixed_patient_split(
    sample_ids: Sequence[str], group_by_sample: Mapping[str, str], seed: int = SEED
) -> FixedSplit:
    """Recreate the legacy sorted-patient, shuffled 70/30 split exactly."""

    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != set(group_by_sample):
        raise ValueError("sample IDs must be unique and have exactly one patient group")
    patients = sorted(set(group_by_sample.values()))
    if len(patients) != 13:
        raise ValueError("legacy Histology split requires exactly 13 patients")
    shuffled = list(patients)
    np.random.default_rng(seed).shuffle(shuffled)
    cut = int(0.7 * len(shuffled))
    train_groups = tuple(sorted(shuffled[:cut]))
    validation_groups = tuple(sorted(shuffled[cut:]))
    train_indices = tuple(i for i, sample in enumerate(sample_ids) if group_by_sample[sample] in train_groups)
    validation_indices = tuple(i for i, sample in enumerate(sample_ids) if group_by_sample[sample] in validation_groups)
    split = FixedSplit(
        train_ids=tuple(sample_ids[i] for i in train_indices),
        validation_ids=tuple(sample_ids[i] for i in validation_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_groups=train_groups,
        validation_groups=validation_groups,
    )
    if set(split.train_groups) & set(split.validation_groups):
        raise RuntimeError("patient groups overlap in fixed split")
    if set(split.train_ids) | set(split.validation_ids) != set(sample_ids):
        raise RuntimeError("fixed split does not cover all ROI samples")
    return split


def compare_binary_metric_records(
    recorded: Mapping[str, Any], observed: Mapping[str, Any], tolerance: float = LEGACY_TOLERANCE
) -> dict[str, Any]:
    """Compare preserved legacy loss/accuracy without rewriting either value."""

    fields: dict[str, Any] = {}
    for field in ("loss", "acc"):
        expected = float(recorded[field])
        actual = float(observed[field])
        difference = abs(expected - actual)
        fields[field] = {
            "recorded": expected,
            "observed": actual,
            "absolute_difference": difference,
            "within_tolerance": difference <= tolerance,
        }
    return {
        **fields,
        "tolerance": tolerance,
        "all_within_tolerance": all(value["within_tolerance"] for value in fields.values()),
    }


def make_prediction_records(
    sample_ids: Sequence[str], labels: Sequence[int], probabilities: Sequence[float], *, branch: str
) -> tuple[PredictionRecord, ...]:
    """Create one finite fixed-threshold prediction per ROI."""

    if not (len(sample_ids) == len(labels) == len(probabilities)):
        raise ValueError("sample IDs, labels, and probabilities must have equal length")
    records = []
    for sample_id, label, probability in zip(sample_ids, labels, probabilities):
        value = float(probability)
        if int(label) not in (0, 1) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("binary predictions require finite probabilities and labels 0/1")
        records.append(
            PredictionRecord(
                key=PredictionKey(sample_id=sample_id, fold_id=FOLD_ID, seed=SEED, inference_branch=branch),
                target_name="tumor",
                y_true=int(label),
                y_pred=int(value >= 0.5),
                probability=value,
            )
        )
    return tuple(records)


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
    for path in (WORKSPACE, DATA_ROOT, GLOBAL_CACHE, LOCAL_CACHE):
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError(f"path must remain on G: {path}")
    if not GLOBAL_CACHE.is_dir() or not LOCAL_CACHE.is_dir():
        raise RuntimeError("required read-only Histology caches are unavailable")


def _discover_rows() -> tuple[tuple[str, str, int], ...]:
    rows = []
    for patient in sorted(DATA_ROOT.glob("P*")):
        if patient.is_dir() and patient.name[1:].isdigit():
            for roi in sorted(patient.glob("ROI_*")):
                label = 1 if roi.name.endswith("_T") and not roi.name.endswith("_NT") else 0
                rows.append((patient.name, roi.name, label))
    if len(rows) != 469:
        raise RuntimeError(f"expected 469 Histology ROIs, observed {len(rows)}")
    return tuple(rows)


def _load_arrays() -> HistologyArrays:
    rgb, global_spectrum, local_tokens, labels, sample_ids = [], [], [], [], []
    hashes: dict[str, dict[str, str]] = {}
    volume = 0
    for patient, roi, label in _discover_rows():
        sample_id = f"{patient}/{roi}"
        global_path = GLOBAL_CACHE / f"{patient}_{roi}.npz"
        local_path = LOCAL_CACHE / f"{patient}_{roi}.npz"
        if not global_path.is_file() or not local_path.is_file():
            raise FileNotFoundError(f"missing read-only cache for {sample_id}")
        hashes[sample_id] = {"global": hash_file(global_path), "local": hash_file(local_path)}
        volume += global_path.stat().st_size + local_path.stat().st_size
        with np.load(global_path, allow_pickle=False) as global_payload, np.load(local_path, allow_pickle=False) as local_payload:
            gv = np.asarray(global_payload["v"], dtype=np.float32)
            gs = np.asarray(global_payload["s"], dtype=np.float32)
            lv = np.asarray(local_payload["v"], dtype=np.float32)
            ls = np.asarray(local_payload["s"], dtype=np.float32)
        if gv.shape != (3, 64, 64) or gs.shape != (128,) or ls.shape != (16, 128):
            raise RuntimeError(f"unexpected representation shape for {sample_id}")
        if lv.shape != gv.shape or not np.array_equal(lv, gv):
            raise RuntimeError(f"global/local RGB cache mismatch for {sample_id}")
        if not np.isfinite(gv).all() or not np.isfinite(gs).all() or not np.isfinite(ls).all():
            raise RuntimeError(f"non-finite representation for {sample_id}")
        rgb.append(gv); global_spectrum.append(gs); local_tokens.append(ls)
        labels.append(label); sample_ids.append(sample_id)
    ids = tuple(sample_ids)
    return HistologyArrays(
        rgb=np.stack(rgb), global_spectrum=np.stack(global_spectrum), local_tokens=np.stack(local_tokens),
        labels=np.asarray(labels, dtype=np.int64), sample_ids=ids,
        group_by_sample={sample: sample.split("/", 1)[0] for sample in ids},
        cache_hashes=hashes, input_volume_bytes=volume,
    )


def _metric_schema() -> MetricSchema:
    return MetricSchema(task_kind=TaskKind.BINARY_CLASSIFICATION, numeric_tolerance=1e-12, decision_threshold=0.5, ece_bins=10)


def _set_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _legacy_model(spec: NeuralSpec) -> torch.nn.Module:
    from src import hypercross_routeA as legacy
    if spec.name == "local_route_a":
        return legacy.MILLocalRouteA(2)
    if spec.name == "mil_concat":
        return legacy.MILConcat(2)
    return legacy.CrossModalRouteA("cls", 2)


def _legacy_metrics(labels: Tensor, logits: Tensor, *, balanced: bool) -> dict[str, Any]:
    total = 0.0
    for start in range(0, len(labels), BATCH_SIZE * 2):
        end = min(start + BATCH_SIZE * 2, len(labels))
        weight = None
        if balanced:
            counts = torch.bincount(labels[start:end], minlength=2).float().clamp_min(1)
            weight = (counts.sum() / (2 * counts)).to(logits.device)
        total += float(F.cross_entropy(logits[start:end], labels[start:end], weight=weight).item()) * (end - start)
    return {"loss": total / len(labels), "acc": float((logits.argmax(1) == labels).float().mean()), "n": len(labels)}


def _legacy_comparisons(arrays: HistologyArrays, split: FixedSplit, device: torch.device, store: FailureAtomicArtifactStore, execution_id: str) -> dict[str, Any]:
    index = np.asarray(split.validation_indices)
    rgb = torch.from_numpy(arrays.rgb[index]).to(device)
    local = torch.from_numpy(arrays.local_tokens[index]).to(device)
    global_spectrum = torch.from_numpy(arrays.global_spectrum[index]).to(device)
    labels = torch.from_numpy(arrays.labels[index]).to(device)
    loader = ExternalCheckpointLoader()
    writer = PredictionArtifactWriter(store)
    results: dict[str, Any] = {}
    for spec in NEURAL_SPECS:
        checkpoint = WORKSPACE / spec.legacy_checkpoint
        history_path = WORKSPACE / spec.legacy_history
        before = hash_file(checkpoint)
        payload = loader.load(checkpoint, source_reference=f"legacy-g-drive://{spec.legacy_checkpoint}", expected_content_hash=spec.legacy_checkpoint_hash, map_location=device)
        model = _legacy_model(spec).to(device=device, dtype=torch.float32)
        model.load_state_dict(payload["model"], strict=True); model.eval()
        with torch.no_grad():
            logits = model(rgb, local if spec.local_input else global_spectrum)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
        observed = _legacy_metrics(labels, logits, balanced=spec.balanced)
        checkpoint_recorded = payload.get("metrics", {})
        comparison = compare_binary_metric_records(checkpoint_recorded, observed)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        recorded_best = max(history, key=lambda row: float(row["acc"]))
        published = writer.publish_shards(
            artifact_id=f"{execution_id}-legacy-{spec.name}-predictions",
            run_id=f"legacy-comparison-{execution_id}-{spec.name}",
            expected_sample_ids=split.validation_ids,
            shards=(make_prediction_records(split.validation_ids, arrays.labels[index], probabilities.cpu().tolist(), branch=f"legacy-{spec.inference_branch}-checkpoint"),),
            relative_path=f"executions/{execution_id}/legacy-comparison/{spec.name}.predictions.canonical.json",
        )
        caveats = ["legacy checkpoint preprocessing lineage is incomplete; this comparison is not formal v2 evidence"]
        if not comparison["all_within_tolerance"]:
            caveats.append("reloaded legacy checkpoint metrics differ from checkpoint payload beyond the declared tolerance")
        results[spec.name] = {
            "model_key": spec.model_key, "checkpoint_reference": spec.legacy_checkpoint,
            "checkpoint_sha256": before, "checkpoint_unchanged": before == hash_file(checkpoint),
            "history_reference": spec.legacy_history, "history_sha256": hash_file(history_path),
            "checkpoint_recorded_epoch": int(payload.get("epoch", recorded_best["epoch"])),
            "history_first_max_accuracy_epoch": int(recorded_best["epoch"]),
            "comparison": comparison, "prediction_artifact": published.reference.model_dump(mode="json"),
            "caveats": caveats,
        }
    return results


def _claim(kind: ImplementationKind) -> ClaimDecision:
    return ClaimDecision(
        status=ClaimStatus.EXPLORATORY,
        supporting_reasons=("task_15_11_unified_runner", "patient_grouped_fixed_split", "prediction_first_metrics"),
        blocking_reasons=("fixed_9_4_development_evidence", "single_seed", "validation_used_for_epoch_selection", "legacy_cache_creation_lineage_incomplete"),
        applied_ceilings=(ClaimCeiling(status=ClaimStatus.EXPLORATORY, reason_code="historical_fixed_9_4_split", evidence_reference=PROTOCOL_KEY),),
        prerequisite_checks={"formal_outer_evaluation": False, "nested_or_lopo_started": False, "single_process": True, "float32_template": True, "classic_model": kind is ImplementationKind.CLASSIC},
    )


def _register_artifact(registry: RunRegistry, reference: Any, kind: str, run_id: str | None) -> None:
    registry.append_artifact(ArtifactRecord(artifact=reference, artifact_kind=kind, run_id=run_id, created_utc=datetime.now(timezone.utc)))


def _publish_metadata(store: FailureAtomicArtifactStore, registry: RunRegistry, execution_id: str, arrays: HistologyArrays, split: FixedSplit, environment: Any, template: Any) -> tuple[str, str, dict[str, Any]]:
    dataset = {
        "schema_version": "1.0.0", "dataset_key": "histology-hsi-gb", "sample_count": len(arrays.sample_ids),
        "sample_ids": arrays.sample_ids, "group_by_sample_id": arrays.group_by_sample,
        "labels": {sample: int(arrays.labels[i]) for i, sample in enumerate(arrays.sample_ids)},
        "cache_content_hashes": arrays.cache_hashes,
        "cache_lineage_status": "legacy_creation_lineage_incomplete_new_content_hash_binding_only",
    }
    dataset_hash = hash_object(dataset)
    protocol = {
        "schema_version": "1.0.0", "protocol_key": PROTOCOL_KEY, "dataset_hash": dataset_hash, "fold_id": FOLD_ID,
        "train_ids": split.train_ids, "validation_ids": split.validation_ids,
        "train_group_ids": split.train_groups, "validation_group_ids": split.validation_groups,
        "split_algorithm": "sorted patient IDs; numpy.default_rng(42).shuffle; floor(0.7*13)",
        "evidence_scope": "fixed-split development evidence with exploratory claim ceiling",
        "epoch_selection": "maximum validation accuracy; first maximum wins",
        "formal_nested_or_lopo_evaluation_budget_consumed": 0,
    }
    protocol_hash = hash_object(protocol)
    refs: dict[str, Any] = {}
    for name, payload in (("dataset", {**dataset, "dataset_hash": dataset_hash}), ("protocol", {**protocol, "protocol_hash": protocol_hash}), ("environment", environment.model_dump(mode="json")), ("template-snapshot", template.model_dump(mode="json"))):
        ref = store.publish_bytes(artifact_id=f"{execution_id}-{name}", relative_path=f"executions/{execution_id}/metadata/{name}.json", content=_json_bytes(payload), media_type="application/json")
        _register_artifact(registry, ref, f"baseline_{name}", None); refs[name] = ref.model_dump(mode="json")
    return dataset_hash, protocol_hash, refs


def _config(template: Any, model_key: str, input_volume: int, parameters: Mapping[str, Any]) -> FrozenRunConfig:
    return FrozenRunConfig.create(model_key=model_key, protocol_key=PROTOCOL_KEY, execution_mode=ExecutionMode.DETERMINISTIC, numeric_precision=NumericPrecision.FLOAT32, framework_settings=template.framework_settings, input_volume_bytes=input_volume, checkpoint_retention=template.checkpoint_retention, resource_budget=template.resource_budget, parameters={"seed": SEED, "fold_id": FOLD_ID, "batch_size": BATCH_SIZE, "formal_evaluation_allowed": False, **dict(parameters)})


def _allocate(registry: RunRegistry, registration: Any, config: FrozenRunConfig, dataset_hash: str, protocol_hash: str, environment_hash: str, source_hash: str, notes: Sequence[str]) -> tuple[RunRecord, FrozenRunIdentity]:
    identity = FrozenRunIdentity(dataset_hash=dataset_hash, protocol_hash=protocol_hash, model_key=registration.model_key, config_hash=config.config_hash, fold_id=FOLD_ID, seed=SEED, environment_lock_hash=environment_hash, environment_compatibility_policy="exact")
    planned = registry.allocate_run(
        frozen_identity=identity, model_identity=registration.identity, input_contract=registration.input_contract,
        provenance=Provenance(code_revision=registration.source_revision, source_hash=source_hash, environment_hash=environment_hash, config_hash=config.config_hash, dataset_hash=dataset_hash, adaptation_notes=tuple(notes)),
        resource_budget=config.resource_budget, claim_decision=_claim(registration.identity.implementation_kind),
    )
    registry.append_run(RunRecord.model_validate({**planned.model_dump(mode="python"), "status": RunStatus.RUNNING}))
    return planned, identity


def _publish(store: FailureAtomicArtifactStore, registry: RunRegistry, execution_id: str, run_id: str, name: str, content: bytes, media_type: str, kind: str) -> Any:
    ref = store.publish_bytes(artifact_id=f"{execution_id}-{run_id}-{name}", relative_path=f"executions/{execution_id}/runs/{run_id}/{name}", content=content, media_type=media_type)
    _register_artifact(registry, ref, kind, run_id); return ref


def _replace_deterministic_pools(model: torch.nn.Module, spec: NeuralSpec) -> tuple[str, ...]:
    changes = []
    legacy = getattr(model, "model", model)
    if spec.name == "local_route_a":
        pool = legacy.v[-1]
        if not isinstance(pool, torch.nn.AdaptiveAvgPool2d) or pool.output_size != (4, 4):
            raise RuntimeError("Local Route A pooling contract changed")
        legacy.v[-1] = torch.nn.AvgPool2d(4, 4); changes.append("AdaptiveAvgPool2d((4,4))->AvgPool2d(4,4)")
    if spec.name == "global_route_a":
        pool = legacy.se.pool
        if not isinstance(pool, torch.nn.AdaptiveAvgPool1d) or pool.output_size != 16:
            raise RuntimeError("global Route A pooling contract changed")
        legacy.se.pool = torch.nn.Identity(); changes.append("AdaptiveAvgPool1d(16)->Identity for length-16 input")
    return tuple(changes)


def _classification_loss(spec: NeuralSpec, model: torch.nn.Module, logits: Tensor, labels: Tensor, visual: Tensor, spectral: Tensor) -> Tensor:
    weight = None
    if spec.balanced:
        counts = torch.bincount(labels, minlength=2).float().clamp_min(1)
        weight = counts.sum() / (2 * counts)
    task = F.cross_entropy(logits, labels, weight=weight)
    if spec.align_weight == 0 and spec.nce_weight == 0:
        return task
    visual = F.normalize(visual, dim=1); spectral = F.normalize(spectral, dim=1)
    alignment = (1 - (visual * spectral).sum(1)).mean()
    legacy = getattr(model, "model", model)
    temperature = legacy.temp.exp().clamp(0.03, 0.5)
    similarity = visual @ spectral.t() / temperature
    targets = torch.arange(len(labels), device=labels.device)
    nce = (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.t(), targets)) / 2
    return task + spec.align_weight * alignment + spec.nce_weight * nce


def _augment(rgb: Tensor, spectrum: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    rgb = rgb.clone(); spectrum = spectrum.clone()
    for index in range(len(rgb)):
        if bool(torch.rand((), generator=generator) < 0.5): rgb[index] = rgb[index].flip(-1)
        if bool(torch.rand((), generator=generator) < 0.5): rgb[index] = rgb[index].flip(-2)
    rgb = (rgb + torch.randn(rgb.shape, generator=generator, dtype=rgb.dtype) * 0.012).clamp(0, 1)
    spectrum = spectrum + torch.randn(spectrum.shape, generator=generator, dtype=spectrum.dtype) * 0.015
    return rgb, spectrum


def _train_neural(spec: NeuralSpec, registry_for_model: Any, arrays: HistologyArrays, split: FixedSplit, dataset_hash: str, protocol_hash: str, environment: Any, template: Any, source_hash: str, store: FailureAtomicArtifactStore, registry: RunRegistry, recovery: RecoveryManager, execution_id: str, device: torch.device) -> dict[str, Any]:
    registration = registry_for_model.resolve(spec.model_key)
    config = _config(template, spec.model_key, arrays.input_volume_bytes, {"epochs": spec.epochs, "optimizer": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4, "alignment_weight": spec.align_weight, "contrastive_weight": spec.nce_weight, "balanced_batch_loss": spec.balanced, "augmentation": True})
    model = registry_for_model.build(spec.model_key)
    registry_for_model.run_metadata(spec.model_key, model, config_hash=config.config_hash, supplied_contract=registration.input_contract.model_copy(deep=True))
    planned, identity = _allocate(registry, registration, config, dataset_hash, protocol_hash, environment.lock_hash, source_hash, ("task-15.11-governed-retraining", "read-only-legacy-cache-content-rebound", "fixed-split-development-evidence"))
    identity_ref = _publish(store, registry, execution_id, planned.run_id, "frozen_run_identity.json", _json_bytes(identity.model_dump(mode="json")), "application/json", "frozen_run_identity")
    pool_changes = _replace_deterministic_pools(model, spec)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, spec.epochs)
    train_index = np.asarray(split.train_indices); validation_index = np.asarray(split.validation_indices)
    spectral_array = arrays.local_tokens if spec.local_input else arrays.global_spectrum
    dataset = TensorDataset(torch.from_numpy(arrays.rgb[train_index]), torch.from_numpy(spectral_array[train_index]), torch.from_numpy(arrays.labels[train_index]))
    loader_generator = torch.Generator().manual_seed(SEED)
    augment_generator = torch.Generator().manual_seed(SEED + 15011)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, generator=loader_generator)
    val_rgb = torch.from_numpy(arrays.rgb[validation_index]).to(device)
    val_spectrum = torch.from_numpy(spectral_array[validation_index]).to(device)
    val_labels = torch.from_numpy(arrays.labels[validation_index]).to(device)
    started = time.monotonic_ns(); torch.cuda.reset_peak_memory_stats(0)
    best_accuracy = -math.inf; best_epoch = 0; best_state = b""; history = []; steps = 0; latest_recovery = None
    try:
        for epoch in range(1, spec.epochs + 1):
            model.train(); total = 0.0; seen = 0
            for rgb, spectral, labels in loader:
                rgb, spectral = _augment(rgb, spectral, augment_generator)
                rgb = rgb.to(device, non_blocking=True); spectral = spectral.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, visual, spectral_features = model(rgb, spectral, return_aux=True)
                loss = _classification_loss(spec, model, logits, labels, visual, spectral_features)
                if not bool(torch.isfinite(loss).item()): raise RuntimeError("non-finite Histology baseline loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
                total += float(loss.detach().item()) * len(labels); seen += len(labels); steps += 1
            scheduler.step(); model.eval()
            with torch.no_grad(): val_logits = model(val_rgb, val_spectrum)
            observed = _legacy_metrics(val_labels, val_logits, balanced=spec.balanced)
            history.append({"epoch": epoch, "train_loss": total / seen, **observed})
            state = _torch_bytes(model.state_dict())
            if observed["acc"] > best_accuracy:
                best_accuracy = observed["acc"]; best_epoch = epoch; best_state = state
            latest_recovery = recovery.save_checkpoint_boundary(
                run_id=planned.run_id, recovery_id=f"{execution_id}-{planned.run_id}-recovery-{epoch}",
                relative_path=f"executions/{execution_id}/runs/{planned.run_id}/recovery/epoch-{epoch}.canonical.json",
                model_state=state, optimizer_state=_torch_bytes(optimizer.state_dict()), scheduler_state=_torch_bytes(scheduler.state_dict()),
                stage_state=_json_bytes({"completed_epoch": epoch, "completed_training_steps": steps, "best_epoch": best_epoch, "best_accuracy": best_accuracy}),
                last_completed_stage=f"epoch_{epoch}", completed_checkpoint_boundary=epoch,
            )
            measurements = RuntimeMeasurements(completed_epochs=epoch, completed_training_steps=steps, wall_clock_duration_ms=(time.monotonic_ns()-started)//1_000_000, peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)), device_identity=environment.device_identity, numeric_precision="float32", parameter_count=registration.identity.parameter_count, input_volume_bytes=arrays.input_volume_bytes)
            if assess_resource_budget(config.resource_budget, measurements, checkpoint_boundary=epoch).hard_exceeded:
                raise RuntimeError("hard resource limit reached at neural checkpoint boundary")
    except Exception:
        recovery.record_interruption(planned.run_id, status=RunStatus.FAILED, last_completed_stage=f"epoch_{len(history)}")
        raise
    selected_ref = _publish(store, registry, execution_id, planned.run_id, "checkpoints/selected.pt", best_state, "application/vnd.pytorch.state-dict", "baseline_selected_checkpoint")
    last_ref = _publish(store, registry, execution_id, planned.run_id, "checkpoints/last.pt", _torch_bytes(model.state_dict()), "application/vnd.pytorch.state-dict", "baseline_last_checkpoint")
    reloaded = registry_for_model.build(spec.model_key); _replace_deterministic_pools(reloaded, spec); reloaded = reloaded.to(device=device, dtype=torch.float32)
    reloaded.load_state_dict(torch.load(io.BytesIO(best_state), map_location=device, weights_only=True), strict=True); reloaded.eval()
    with torch.no_grad(): probabilities = torch.softmax(reloaded(val_rgb, val_spectrum), dim=1)[:, 1].cpu().tolist()
    published = PredictionArtifactWriter(store).publish_shards(
        artifact_id=f"{execution_id}-{planned.run_id}-predictions", run_id=planned.run_id, expected_sample_ids=split.validation_ids,
        shards=(make_prediction_records(split.validation_ids, arrays.labels[validation_index], probabilities, branch=spec.inference_branch),),
        relative_path=f"executions/{execution_id}/runs/{planned.run_id}/predictions.canonical.json",
    )
    _register_artifact(registry, published.reference, "baseline_prediction", planned.run_id)
    metrics = binary_metrics(published.artifact, _metric_schema())
    final_measurements = RuntimeMeasurements(completed_epochs=spec.epochs, completed_training_steps=steps, wall_clock_duration_ms=(time.monotonic_ns()-started)//1_000_000, peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)), device_identity=environment.device_identity, numeric_precision="float32", parameter_count=registration.identity.parameter_count, input_volume_bytes=arrays.input_volume_bytes)
    assessment = assess_resource_budget(config.resource_budget, final_measurements, checkpoint_boundary=spec.epochs)
    caveats = ("fixed 9/4 patient split is development evidence with an exploratory claim ceiling", "validation labels selected the epoch", "single seed 42", "legacy cache creation lineage is incomplete; exact cache bytes are newly bound only", *(pool_changes or ()))
    running = registry.latest_run(planned.run_id)
    complete = RunRecord.model_validate({**running.model_dump(mode="python"), "status": RunStatus.COMPLETE, "checkpoint_ref": selected_ref, "recovery_artifact_ref": latest_recovery.artifact, "prediction_artifact_ref": published.reference, "metrics": metrics.model_dump(mode="json"), "runtime_measurements": final_measurements, "completed_utc": datetime.now(timezone.utc)})
    registry.append_run(complete)
    return {"run_id": planned.run_id, "model_key": spec.model_key, "selected_epoch": best_epoch, "selection_accuracy": best_accuracy, "metrics": metrics.model_dump(mode="json"), "prediction_artifact": published.reference.model_dump(mode="json"), "selected_checkpoint": selected_ref.model_dump(mode="json"), "last_checkpoint": last_ref.model_dump(mode="json"), "recovery_artifact": latest_recovery.artifact.model_dump(mode="json"), "identity_artifact": identity_ref.model_dump(mode="json"), "runtime_measurements": final_measurements.model_dump(mode="json"), "budget_assessment": assessment.model_dump(mode="json"), "caveats": list(caveats)}


def _train_classic(model_key: str, branch: str, model: Any, registration: Any, arrays: HistologyArrays, split: FixedSplit, dataset_hash: str, protocol_hash: str, environment: Any, template: Any, source_hash: str, store: FailureAtomicArtifactStore, registry: RunRegistry, recovery: RecoveryManager, execution_id: str) -> dict[str, Any]:
    parameters = {"class_weight": "balanced", "probability": True} if model_key == SVM_MODEL_KEY else {"estimators": 300, "class_weight": "balanced", "n_jobs": 1}
    config = _config(template, model_key, arrays.input_volume_bytes, parameters)
    planned, identity = _allocate(registry, registration, config, dataset_hash, protocol_hash, environment.lock_hash, source_hash, ("task-15.11-governed-classic-fit", "read-only-legacy-cache-content-rebound", "fixed-split-development-evidence"))
    identity_ref = _publish(store, registry, execution_id, planned.run_id, "frozen_run_identity.json", _json_bytes(identity.model_dump(mode="json")), "application/json", "frozen_run_identity")
    train_index = np.asarray(split.train_indices); validation_index = np.asarray(split.validation_indices)
    started = time.monotonic_ns()
    try:
        model.fit(arrays.global_spectrum[train_index], arrays.labels[train_index])
        model_bytes = pickle.dumps(model, protocol=5)
        recovery_ref = recovery.save_checkpoint_boundary(run_id=planned.run_id, recovery_id=f"{execution_id}-{planned.run_id}-recovery-1", relative_path=f"executions/{execution_id}/runs/{planned.run_id}/recovery/fit-boundary.canonical.json", model_state=model_bytes, optimizer_state=_json_bytes({"not_applicable": True}), scheduler_state=_json_bytes({"not_applicable": True}), stage_state=_json_bytes({"completed_fit": True, "completed_training_steps": 1}), last_completed_stage="classic_fit", completed_checkpoint_boundary=1)
    except Exception:
        recovery.record_interruption(planned.run_id, status=RunStatus.FAILED, last_completed_stage="classic_fit_failed"); raise
    selected = _publish(store, registry, execution_id, planned.run_id, "checkpoints/selected.pkl", model_bytes, "application/octet-stream", "baseline_selected_checkpoint")
    last = _publish(store, registry, execution_id, planned.run_id, "checkpoints/last.pkl", model_bytes, "application/octet-stream", "baseline_last_checkpoint")
    restored = pickle.loads(model_bytes)
    probabilities = restored.predict_proba(arrays.global_spectrum[validation_index])[:, 1]
    published = PredictionArtifactWriter(store).publish_shards(artifact_id=f"{execution_id}-{planned.run_id}-predictions", run_id=planned.run_id, expected_sample_ids=split.validation_ids, shards=(make_prediction_records(split.validation_ids, arrays.labels[validation_index], probabilities, branch=branch),), relative_path=f"executions/{execution_id}/runs/{planned.run_id}/predictions.canonical.json")
    _register_artifact(registry, published.reference, "baseline_prediction", planned.run_id)
    metrics = binary_metrics(published.artifact, _metric_schema())
    measurements = RuntimeMeasurements(completed_epochs=1, completed_training_steps=1, wall_clock_duration_ms=(time.monotonic_ns()-started)//1_000_000, peak_device_memory_bytes=0, device_identity=environment.device_identity, numeric_precision="float32", parameter_count=0, input_volume_bytes=arrays.input_volume_bytes)
    assessment = assess_resource_budget(config.resource_budget, measurements, checkpoint_boundary=1)
    caveats = ("fixed 9/4 patient split is development evidence with an exploratory claim ceiling", "single seed 42", "classic control executes on CPU within the TCFM single process", "legacy cache creation lineage is incomplete; exact cache bytes are newly bound only")
    running = registry.latest_run(planned.run_id)
    registry.append_run(RunRecord.model_validate({**running.model_dump(mode="python"), "status": RunStatus.COMPLETE, "checkpoint_ref": selected, "recovery_artifact_ref": recovery_ref.artifact, "prediction_artifact_ref": published.reference, "metrics": metrics.model_dump(mode="json"), "runtime_measurements": measurements, "completed_utc": datetime.now(timezone.utc)}))
    return {"run_id": planned.run_id, "model_key": model_key, "metrics": metrics.model_dump(mode="json"), "prediction_artifact": published.reference.model_dump(mode="json"), "selected_checkpoint": selected.model_dump(mode="json"), "last_checkpoint": last.model_dump(mode="json"), "recovery_artifact": recovery_ref.artifact.model_dump(mode="json"), "identity_artifact": identity_ref.model_dump(mode="json"), "runtime_measurements": measurements.model_dump(mode="json"), "budget_assessment": assessment.model_dump(mode="json"), "caveats": list(caveats)}


def _close_orphaned_runs(registry: RunRegistry, recovery: RecoveryManager) -> tuple[str, ...]:
    """Close active append-only records left by a terminated prior attempt."""

    if not registry.path.exists():
        return ()
    latest: dict[str, dict[str, Any]] = {}
    for line in registry.path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("entry_kind") == "run":
            record = entry["record"]
            latest[str(record["run_id"])] = record
    orphaned = tuple(sorted((run_id for run_id, record in latest.items() if record.get("status") == RunStatus.RUNNING.value), key=lambda value: value.encode("utf-8")))
    for run_id in orphaned:
        recovery.record_interruption(run_id, status=RunStatus.FAILED, last_completed_stage="prior_attempt_failed_after_last_recovery_boundary")
    return orphaned


def execute_baseline() -> dict[str, Any]:
    _validate_roots(); template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.BASELINE: raise RuntimeError("task 15.11 requires the immutable baseline template")
    environment = validate_live_environment(template); _set_determinism(); torch.cuda.set_device(0); device = torch.device("cuda:0")
    execution_id = datetime.now(timezone.utc).strftime("task-15-11-%Y%m%dT%H%M%S%fZ")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    store = FailureAtomicArtifactStore(OUTPUT_ROOT); registry = RunRegistry(OUTPUT_ROOT / "registry.jsonl"); recovery = RecoveryManager(registry, store)
    orphaned_run_ids = _close_orphaned_runs(registry, recovery)
    protected_roots = {"src": WORKSPACE / "src", "runs": WORKSPACE / "runs", "dataset": DATA_ROOT}
    readonly_before = {name: _tree_fingerprint(path) for name, path in protected_roots.items()}
    arrays = _load_arrays(); split = fixed_patient_split(arrays.sample_ids, arrays.group_by_sample)
    expected_train = ("P1", "P11", "P12", "P2", "P3", "P4", "P6", "P7", "P9")
    expected_validation = ("P10", "P13", "P5", "P8")
    if split.train_groups != expected_train or split.validation_groups != expected_validation or len(split.train_ids) != 321 or len(split.validation_ids) != 148:
        raise RuntimeError("legacy fixed 9/4 patient split recreation failed")
    dataset_hash, protocol_hash, metadata_refs = _publish_metadata(store, registry, execution_id, arrays, split, environment, template)
    legacy = _legacy_comparisons(arrays, split, device, store, execution_id)
    candidate_registry = create_histology_model_registry(); global_registry = create_histology_baseline_registry()
    neural_source_hash = hash_object({"runner": hash_file(Path(__file__)), "histology_adapters": hash_file(WORKSPACE / "benchmark/models/histology.py"), "global_adapter": hash_file(WORKSPACE / "benchmark/models/histology_baseline.py"), "legacy_source": hash_file(WORKSPACE / "src/hypercross_routeA.py")})
    classic_source_hash = hash_object({"runner": hash_file(Path(__file__)), "histology_adapters": hash_file(WORKSPACE / "benchmark/models/histology.py"), "sklearn_version": environment.sklearn_version})
    runs = []
    for spec in NEURAL_SPECS:
        runs.append(_train_neural(spec, global_registry if spec.name == "global_route_a" else candidate_registry, arrays, split, dataset_hash, protocol_hash, environment, template, neural_source_hash, store, registry, recovery, execution_id, device))
    for model_key, branch, cls in ((SVM_MODEL_KEY, SVM_INFERENCE_BRANCH, HistologySVMAdapter), (RF_MODEL_KEY, RF_INFERENCE_BRANCH, HistologyRandomForestAdapter)):
        registration = candidate_registry.resolve(model_key); model = cls(seed=SEED)
        candidate_registry.run_metadata(model_key, model, config_hash="0" * 64, supplied_contract=registration.input_contract.model_copy(deep=True))
        runs.append(_train_classic(model_key, branch, model, registration, arrays, split, dataset_hash, protocol_hash, environment, template, classic_source_hash, store, registry, recovery, execution_id))
    readonly_after = {name: _tree_fingerprint(path) for name, path in protected_roots.items()}
    if readonly_before != readonly_after: raise RuntimeError("read-only src, runs, or Histology dataset/cache metadata changed")
    mismatches = [name for name, item in legacy.items() if not item["comparison"]["all_within_tolerance"]]
    report = {
        "schema_version": "1.0.0", "task_id": "15.11", "status": "pass", "execution_id": execution_id,
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": "conda run --no-capture-output -n TCFM python -m benchmark.runners.histology_baseline --execute",
        "scope": {"dataset_key": "histology-hsi-gb", "output_root": "runs_v2/baselines/histology", "evidence_scope": "fixed-split development evidence", "claim_ceiling": "exploratory", "src_runs_dataset_caches_read_only": True, "nested_cv_started": False, "lopo_started": False, "formal_outer_evaluation_budget_consumed": 0, "process_count": 1},
        "environment": environment.model_dump(mode="json"),
        "template": {"reference": "configs/fast_lane/baseline.json", "content_hash": hash_file(TEMPLATE_PATH), "template_hash": template.template_hash, "numeric_precision": template.numeric_precision.value, "execution_mode": template.execution_mode.value, "framework_settings": template.framework_settings.model_dump(mode="json"), "resource_budget": template.resource_budget.model_dump(mode="json")},
        "identity": {"dataset_hash": dataset_hash, "protocol_hash": protocol_hash, "protocol_key": PROTOCOL_KEY, "fold_id": FOLD_ID, "neural_model_source_hash": neural_source_hash, "classic_model_source_hash": classic_source_hash, "metadata_artifacts": metadata_refs},
        "split": {"train_count": len(split.train_ids), "validation_count": len(split.validation_ids), "train_group_ids": split.train_groups, "validation_group_ids": split.validation_groups, "groups_disjoint": True, "seed": SEED},
        "legacy_tolerance_comparisons": legacy, "legacy_mismatches": mismatches, "prior_orphaned_runs_closed_as_failed": list(orphaned_run_ids), "governed_runs": runs,
        "caveats": [
            "The historical fixed 9/4 patient split is development evidence and retains the spec-mandated exploratory claim ceiling.",
            "Validation labels selected neural epochs, so these runs are not nested or strict outer evaluations.",
            "Legacy checkpoints lack complete preprocessing lineage; reload comparisons are tolerance checks only and are not formal v2 evidence.",
            "The governed runs bind exact legacy cache bytes but cannot reconstruct or retroactively prove the caches' original fitted-sample lineage.",
            "Only seed 42 was run. SVM and Random Forest execute on CPU in the same TCFM process; neural models execute on cuda:0.",
            "Deterministic CUDA-compatible pooling substitutions are parameter-free and are recorded per affected run.",
            *([f"Legacy tolerance mismatch recorded for: {', '.join(mismatches)}"] if mismatches else []),
        ],
    }
    report_ref = store.publish_bytes(artifact_id=f"{execution_id}-report", relative_path=f"executions/{execution_id}/task-15.11-report.json", content=_json_bytes(report), media_type="application/json")
    _register_artifact(registry, report_ref, "task_15_11_report", None)
    return {"status": "pass", "execution_id": execution_id, "report": report_ref.model_dump(mode="json"), "run_ids": [item["run_id"] for item in runs], "legacy_all_within_tolerance": not mismatches, "legacy_mismatches": mismatches}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute: parser.error("--execute is required")
    try: result = execute_baseline()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, sort_keys=True), file=__import__("sys").stderr); return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
