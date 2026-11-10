"""Governed seed-42 inner-patient screening for staged task 15.12.

The executor uses the 25 inner folds nested inside the five immutable
``hist.patient-nested5.v1`` outer folds.  Candidate, epoch, threshold, and
calibration selection never read a sample while it is an outer-test member.
No strict outer evaluation or LOPO evaluation is available from this module.
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
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark.cli_runtime import validate_live_environment
from benchmark.adapters.histology import (
    CACHED_SPATIAL_MEAN_PCA_KEY,
    FULL_CUBE_SPATIAL_PCA_KEY,
    RGB_PREVIEW_KEY,
)
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import binary_metrics
from benchmark.models.histology import (
    ATTENTION_DISABLED_INFERENCE_BRANCH,
    ATTENTION_DISABLED_MODEL_KEY,
    LOCAL_ROUTE_A_INFERENCE_BRANCH,
    LOCAL_ROUTE_A_MODEL_KEY,
    MIL_CONCAT_INFERENCE_BRANCH,
    MIL_CONCAT_MODEL_KEY,
    PCA_ONLY_INFERENCE_BRANCH,
    PCA_ONLY_MODEL_KEY,
    RGB_ONLY_INFERENCE_BRANCH,
    RGB_ONLY_MODEL_KEY,
    create_histology_model_registry,
)
from benchmark.protocols import ProtocolPlanner
from benchmark.protocols.histology_nested import HistologyNested5Planner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.runners.histology_baseline import HistologyArrays, _load_arrays, _tree_fingerprint
from benchmark.runners.histology_candidates import (
    ATTENTION_DISABLED_CONTROL_KEY,
    LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
    LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
    LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
    LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
    MIL_CONCAT_CONTROL_KEY,
    PCA_TASK_ONLY_CONTROL_KEY,
)
from benchmark.runners.histology_statistics import HistologyStatisticsWiring
from benchmark.schema import (
    MetricSchema,
    OuterFold,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    RuntimeMeasurements,
    SampleManifest,
    TaskKind,
    assess_resource_budget,
    hash_file,
    hash_object,
)


WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\PKG - HistologyHSI-GB")
MANIFEST_PATH = WORKSPACE / "artifacts" / "manifests" / "histology-hsi-gb" / "ba6a973ffa80636d42d50e274719bb72b4907766056b63e90d078da472704e12.json"
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "screening.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "screening" / "histology"
PROTOCOL_KEY = "hist.patient-nested5.inner-screening.v1"
SEED = 42
BATCH_SIZE = 16
EVALUATION_BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
RUNG_EPOCHS = (10, 20, 40, 80)
RUNG_SURVIVORS = (4, 2, 1, 1)
PRIMARY_THRESHOLD = 0.5


@dataclass(frozen=True)
class ScreeningCandidate:
    candidate_id: str
    model_key: str
    inference_branch: str
    role: str
    representation_key: str
    training_recipe_key: str
    loss_key: str
    alignment_weight: float = 0.0
    nce_weight: float = 0.0
    required_control_keys: tuple[str, ...] = ()

    @property
    def safe_id(self) -> str:
        return hashlib.sha256(self.candidate_id.encode("utf-8")).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return hash_object(asdict(self))


@dataclass(frozen=True)
class ScreeningFold:
    outer_fold_id: str
    fold_id: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    forbidden_outer_test_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]


@dataclass
class FoldTrainingState:
    fold: ScreeningFold
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    completed_steps: int = 0


def build_screening_candidates() -> tuple[ScreeningCandidate, ...]:
    """Return the complete predeclared task-15.12 comparison set."""

    local_representation = FULL_CUBE_SPATIAL_PCA_KEY
    candidates = (
        ScreeningCandidate(
            PCA_TASK_ONLY_CONTROL_KEY,
            PCA_ONLY_MODEL_KEY,
            PCA_ONLY_INFERENCE_BRANCH,
            "modality_control",
            CACHED_SPATIAL_MEAN_PCA_KEY,
            "pca-only-supervised-training.v1",
            "classification-only.v1",
        ),
        ScreeningCandidate(
            "histology.rgb-only.task-only-control.v1",
            RGB_ONLY_MODEL_KEY,
            RGB_ONLY_INFERENCE_BRANCH,
            "modality_control",
            RGB_PREVIEW_KEY,
            "rgb-only-supervised-training.v1",
            "classification-only.v1",
        ),
        ScreeningCandidate(
            MIL_CONCAT_CONTROL_KEY,
            MIL_CONCAT_MODEL_KEY,
            MIL_CONCAT_INFERENCE_BRANCH,
            "matched_input_control",
            local_representation,
            "mil-concat.v1",
            "classification-alignment-nce.v1",
            alignment_weight=0.08,
            nce_weight=0.04,
        ),
        ScreeningCandidate(
            LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY,
            LOCAL_ROUTE_A_MODEL_KEY,
            LOCAL_ROUTE_A_INFERENCE_BRANCH,
            "candidate",
            local_representation,
            "local-route-a.v1",
            "classification-alignment-nce.v1",
            alignment_weight=0.08,
            nce_weight=0.04,
            required_control_keys=tuple(sorted((
                ATTENTION_DISABLED_CONTROL_KEY,
                LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
                LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
                LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
                MIL_CONCAT_CONTROL_KEY,
            ), key=str.encode)),
        ),
        ScreeningCandidate(
            ATTENTION_DISABLED_CONTROL_KEY,
            ATTENTION_DISABLED_MODEL_KEY,
            ATTENTION_DISABLED_INFERENCE_BRANCH,
            "attention_specific_control",
            local_representation,
            "local-route-a-attention-disabled.v1",
            "classification-alignment-nce.v1",
            alignment_weight=0.08,
            nce_weight=0.04,
        ),
        ScreeningCandidate(
            LOCAL_ROUTE_A_TASK_ONLY_CONTROL_KEY,
            LOCAL_ROUTE_A_MODEL_KEY,
            LOCAL_ROUTE_A_INFERENCE_BRANCH,
            "loss_control",
            local_representation,
            "local-route-a.v1",
            "classification-only.v1",
        ),
        ScreeningCandidate(
            LOCAL_ROUTE_A_ALIGNMENT_ONLY_CONTROL_KEY,
            LOCAL_ROUTE_A_MODEL_KEY,
            LOCAL_ROUTE_A_INFERENCE_BRANCH,
            "loss_control",
            local_representation,
            "local-route-a.v1",
            "classification-alignment-only.v1",
            alignment_weight=0.08,
        ),
        ScreeningCandidate(
            LOCAL_ROUTE_A_NCE_ONLY_CONTROL_KEY,
            LOCAL_ROUTE_A_MODEL_KEY,
            LOCAL_ROUTE_A_INFERENCE_BRANCH,
            "loss_control",
            local_representation,
            "local-route-a.v1",
            "classification-nce-only.v1",
            nce_weight=0.04,
        ),
    )
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id.encode("utf-8")))
    if len(ordered) != 8 or len({item.candidate_id for item in ordered}) != 8:
        raise RuntimeError("task 15.12 requires eight unique governed candidates/controls")
    return ordered


def build_screening_folds(protocol_folds: Sequence[OuterFold], group_by_sample: Mapping[str, str]) -> tuple[ScreeningFold, ...]:
    """Flatten exact nested inner folds while retaining each outer exclusion set."""

    result: list[ScreeningFold] = []
    for outer in protocol_folds:
        outer_train = set(outer.outer_train_ids)
        outer_test = set(outer.outer_test_ids)
        observed: list[str] = []
        for inner in outer.inner_folds:
            train = set(inner.train_ids)
            validation = set(inner.validation_ids)
            if train & validation or (train | validation) != outer_train:
                raise RuntimeError(f"invalid inner exact coverage in {inner.fold_id}")
            if (train | validation) & outer_test:
                raise RuntimeError(f"outer-test leakage in {inner.fold_id}")
            train_groups = {group_by_sample[item] for item in train}
            validation_groups = {group_by_sample[item] for item in validation}
            if train_groups & validation_groups:
                raise RuntimeError(f"patient leakage in {inner.fold_id}")
            observed.extend(inner.validation_ids)
            result.append(ScreeningFold(
                outer_fold_id=outer.fold_id,
                fold_id=inner.fold_id,
                train_ids=inner.train_ids,
                validation_ids=inner.validation_ids,
                forbidden_outer_test_ids=outer.outer_test_ids,
                validation_group_ids=tuple(sorted(validation_groups, key=str.encode)),
            ))
        if len(observed) != len(set(observed)) or set(observed) != outer_train:
            raise RuntimeError(f"inner OOF coverage is not exact for {outer.fold_id}")
    if len(result) != 25:
        raise RuntimeError("task 15.12 requires five inner folds for each of five outer folds")
    return tuple(result)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


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
    with path.open("ab") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


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


def _load_manifest() -> SampleManifest:
    manifest = SampleManifest.model_validate(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    if manifest.dataset_hash != MANIFEST_PATH.stem or len(manifest.records) != 469:
        raise RuntimeError("frozen Histology manifest identity is invalid")
    return manifest


def _metric_schema(threshold: float = PRIMARY_THRESHOLD) -> MetricSchema:
    return MetricSchema(
        task_kind=TaskKind.BINARY_CLASSIFICATION,
        numeric_tolerance=1e-12,
        decision_threshold=float(threshold),
        ece_bins=10,
    )


def _artifact_from_rows(rows: Sequence[Mapping[str, Any]], *, artifact_id: str, run_id: str) -> PredictionArtifact:
    records = tuple(PredictionRecord(
        key=PredictionKey(
            sample_id=str(row["sample_id"]),
            fold_id=str(row["fold_id"]),
            seed=SEED,
            inference_branch=str(row["inference_branch"]),
        ),
        target_name="tumor",
        y_true=int(row["y_true"]),
        y_pred=int(float(row["probability"]) >= PRIMARY_THRESHOLD),
        probability=float(row["probability"]),
    ) for row in rows)
    expected = tuple(sorted((record.key.sample_id for record in records), key=str.encode))
    if len(expected) != len(set(expected)):
        raise ValueError("one metric artifact cannot contain repeated Sample_IDs")
    return PredictionArtifact.create(
        artifact_id=artifact_id,
        run_id=run_id,
        records=records,
        expected_sample_ids=expected,
        complete=True,
    )


def _metric_values(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    # Metric artifacts require one fold identity. The original inner-fold identity
    # remains in the authoritative exact-OOF rows; this derived artifact names
    # the aggregation stratum explicitly.
    normalized = [{**row, "fold_id": label} for row in rows]
    artifact = _artifact_from_rows(normalized, artifact_id=f"screening-{label}", run_id=f"screening-{label}")
    result = binary_metrics(artifact, _metric_schema()).model_dump(mode="json")
    return {
        **{name: payload for name, payload in result["metrics"].items()},
        "confusion_matrix": result["confusion_matrices"]["confusion_matrix"],
        "source_prediction_hash": result["source_prediction_hash"],
        "sample_count": len(rows),
    }


def _macro_metric(metric_sets: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    values = [item[name]["value"] for item in metric_sets if item[name]["value"] is not None]
    return {
        "value": None if not values else float(np.mean(np.asarray(values, dtype=np.float64))),
        "defined_count": len(values),
        "total_count": len(metric_sets),
    }


def compute_screening_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute exact per-inner-fold, per-outer-fold, patient, and pooled evidence."""

    if not rows:
        raise ValueError("screening rows cannot be empty")
    keys = [(row["outer_fold_id"], row["fold_id"], row["sample_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("screening prediction keys must be unique")
    per_inner: dict[str, Any] = {}
    for fold_id in sorted({str(row["fold_id"]) for row in rows}, key=str.encode):
        selected = [row for row in rows if row["fold_id"] == fold_id]
        per_inner[fold_id] = _metric_values(selected, label=fold_id)
    per_outer: dict[str, Any] = {}
    for outer_id in sorted({str(row["outer_fold_id"]) for row in rows}, key=str.encode):
        selected = [row for row in rows if row["outer_fold_id"] == outer_id]
        per_outer[outer_id] = _metric_values(selected, label=f"{outer_id}-inner-oof")
    per_patient: dict[str, Any] = {}
    for group_id in sorted({str(row["group_id"]) for row in rows}, key=str.encode):
        selected = [row for row in rows if row["group_id"] == group_id]
        # A patient appears in four outer-train contexts. Average those predictions
        # before patient-level metrics so every ROI contributes exactly once.
        by_sample: dict[str, list[Mapping[str, Any]]] = {}
        for row in selected:
            by_sample.setdefault(str(row["sample_id"]), []).append(row)
        averaged = [{
            **values[0],
            "fold_id": "cross-context-patient-oof",
            "probability": float(np.mean([float(item["probability"]) for item in values])),
        } for _, values in sorted(by_sample.items(), key=lambda item: item[0].encode("utf-8"))]
        per_patient[group_id] = _metric_values(averaged, label=f"patient-{group_id}")
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    averaged_rows = [{
        **values[0],
        "fold_id": "cross-context-inner-oof",
        "probability": float(np.mean([float(item["probability"]) for item in values])),
    } for _, values in sorted(by_sample.items(), key=lambda item: item[0].encode("utf-8"))]
    cross_context = _metric_values(averaged_rows, label="cross-context-inner-oof")
    macro = {
        name: _macro_metric(tuple(per_outer.values()), name)
        for name in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity", "ece")
    }
    return {
        "primary_objective": "outer_fold_macro.inner_oof.balanced_accuracy",
        "outer_fold_macro": macro,
        "cross_context_pooled_oof": cross_context,
        "per_outer_fold_inner_oof": per_outer,
        "per_inner_fold": per_inner,
        "per_patient_cross_context": per_patient,
    }


def _score(metrics: Mapping[str, Any]) -> float:
    value = metrics["outer_fold_macro"]["balanced_accuracy"]["value"]
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError("candidate produced undefined primary balanced accuracy")
    return float(value)


def _replace_deterministic_pools(model: nn.Module, candidate: ScreeningCandidate) -> tuple[str, ...]:
    changes: list[str] = []
    legacy = getattr(model, "model", model)
    if candidate.model_key == PCA_ONLY_MODEL_KEY:
        pool = legacy.e.pool
        if not isinstance(pool, nn.AdaptiveAvgPool1d) or pool.output_size != 16:
            raise RuntimeError("PCA-only pooling contract changed")
        legacy.e.pool = nn.Identity()
        changes.append("AdaptiveAvgPool1d(16)->Identity for length-16 input")
    if candidate.model_key == LOCAL_ROUTE_A_MODEL_KEY:
        pool = legacy.v[-1]
        if not isinstance(pool, nn.AdaptiveAvgPool2d) or pool.output_size != (4, 4):
            raise RuntimeError("Local Route A pooling contract changed")
        legacy.v[-1] = nn.AvgPool2d(4, 4)
        changes.append("AdaptiveAvgPool2d((4,4))->AvgPool2d(4,4)")
    if candidate.model_key == ATTENTION_DISABLED_MODEL_KEY:
        pool = legacy.visual[-1]
        if not isinstance(pool, nn.AdaptiveAvgPool2d) or pool.output_size != (4, 4):
            raise RuntimeError("attention-disabled pooling contract changed")
        legacy.visual[-1] = nn.AvgPool2d(4, 4)
        changes.append("AdaptiveAvgPool2d((4,4))->AvgPool2d(4,4)")
    return tuple(changes)


def _forward(model: nn.Module, candidate: ScreeningCandidate, rgb: Tensor, global_spectrum: Tensor, local_tokens: Tensor, *, return_aux: bool) -> Any:
    if candidate.model_key == PCA_ONLY_MODEL_KEY:
        return model(global_spectrum, return_aux=return_aux)
    if candidate.model_key == RGB_ONLY_MODEL_KEY:
        return model(rgb, return_aux=return_aux)
    return model(rgb, local_tokens, return_aux=return_aux)


def _loss(model: nn.Module, candidate: ScreeningCandidate, logits: Tensor, labels: Tensor, visual: Tensor | None, spectral: Tensor | None) -> Tensor:
    task = F.cross_entropy(logits, labels)
    if candidate.alignment_weight == 0.0 and candidate.nce_weight == 0.0:
        return task
    if visual is None or spectral is None:
        raise RuntimeError("declared auxiliary loss requires visual and spectral embeddings")
    visual = F.normalize(visual, dim=1)
    spectral = F.normalize(spectral, dim=1)
    alignment = (1.0 - (visual * spectral).sum(dim=1)).mean()
    legacy = getattr(model, "model", model)
    temperature = legacy.temp.exp().clamp(0.03, 0.5)
    similarity = visual @ spectral.t() / temperature
    targets = torch.arange(len(labels), device=labels.device)
    nce = (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.t(), targets)) / 2.0
    return task + candidate.alignment_weight * alignment + candidate.nce_weight * nce


def _fold_indices(fold: ScreeningFold, id_to_index: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([id_to_index[item] for item in fold.train_ids], dtype=np.int64),
        np.asarray([id_to_index[item] for item in fold.validation_ids], dtype=np.int64),
    )


def _make_states(candidate: ScreeningCandidate, folds: Sequence[ScreeningFold], registry: Any, device: torch.device) -> tuple[list[FoldTrainingState], tuple[str, ...]]:
    states: list[FoldTrainingState] = []
    pool_changes: tuple[str, ...] = ()
    for fold_index, fold in enumerate(folds):
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(SEED + fold_index)
            torch.cuda.manual_seed_all(SEED + fold_index)
            model = registry.build(candidate.model_key)
        changes = _replace_deterministic_pools(model, candidate)
        if not pool_changes:
            pool_changes = changes
        elif changes != pool_changes:
            raise RuntimeError("deterministic pooling substitutions differ by fold")
        model = model.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RUNG_EPOCHS[-1])
        states.append(FoldTrainingState(fold, model, optimizer, scheduler))
    return states, pool_changes


def _checkpoint_payload(state: FoldTrainingState, candidate: ScreeningCandidate, epoch: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_id": "15.12",
        "candidate_id": candidate.candidate_id,
        "candidate_config_hash": candidate.config_hash,
        "fold_id": state.fold.fold_id,
        "outer_fold_id": state.fold.outer_fold_id,
        "seed": SEED,
        "completed_epoch": epoch,
        "completed_training_steps": state.completed_steps,
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
    }


def _load_resume(candidate_root: Path, states: Sequence[FoldTrainingState], candidate: ScreeningCandidate, device: torch.device) -> tuple[int, list[dict[str, Any]], float, int]:
    marker_path = candidate_root / "common-recovery.json"
    if not marker_path.exists():
        return 0, [], -math.inf, 0
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["candidate_config_hash"] != candidate.config_hash:
        raise RuntimeError("candidate recovery config hash mismatch")
    epoch = int(marker["completed_epoch"])
    for state in states:
        checkpoint = candidate_root / "folds" / state.fold.fold_id / "recovery" / f"slot-{epoch % 2}.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["candidate_config_hash"] != candidate.config_hash or int(payload["completed_epoch"]) != epoch:
            raise RuntimeError("fold recovery boundary differs from common recovery marker")
        state.model.load_state_dict(payload["model"], strict=True)
        state.optimizer.load_state_dict(payload["optimizer"])
        state.scheduler.load_state_dict(payload["scheduler"])
        state.completed_steps = int(payload["completed_training_steps"])
    return epoch, list(marker["history"]), float(marker["best_score"]), int(marker["selected_epoch"])


def _train_one_epoch(state: FoldTrainingState, candidate: ScreeningCandidate, arrays: HistologyArrays, id_to_index: Mapping[str, int], epoch: int, device: torch.device) -> float:
    train_indices, _ = _fold_indices(state.fold, id_to_index)
    order = np.random.default_rng(SEED * 1_000_003 + int(state.fold.fold_id.split("-")[-1]) * 10_007 + epoch).permutation(train_indices)
    state.model.train()
    total = 0.0
    seen = 0
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start + BATCH_SIZE]
        rgb = torch.from_numpy(arrays.rgb[indices]).to(device, non_blocking=True)
        global_spectrum = torch.from_numpy(arrays.global_spectrum[indices]).to(device, non_blocking=True)
        local_tokens = torch.from_numpy(arrays.local_tokens[indices]).to(device, non_blocking=True)
        labels = torch.from_numpy(arrays.labels[indices]).to(device, non_blocking=True)
        state.optimizer.zero_grad(set_to_none=True)
        needs_aux = candidate.alignment_weight > 0.0 or candidate.nce_weight > 0.0
        output = _forward(state.model, candidate, rgb, global_spectrum, local_tokens, return_aux=needs_aux)
        if needs_aux:
            logits, visual, spectral = output
        else:
            logits, visual, spectral = output, None, None
        loss = _loss(state.model, candidate, logits, labels, visual, spectral)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss in {candidate.candidate_id}/{state.fold.fold_id}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), GRADIENT_CLIP)
        state.optimizer.step()
        total += float(loss.detach().item()) * len(indices)
        seen += len(indices)
        state.completed_steps += 1
    state.scheduler.step()
    return total / seen


def _predict(state: FoldTrainingState, candidate: ScreeningCandidate, arrays: HistologyArrays, id_to_index: Mapping[str, int], device: torch.device) -> list[dict[str, Any]]:
    _, validation_indices = _fold_indices(state.fold, id_to_index)
    rows: list[dict[str, Any]] = []
    state.model.eval()
    with torch.no_grad():
        for start in range(0, len(validation_indices), EVALUATION_BATCH_SIZE):
            indices = validation_indices[start:start + EVALUATION_BATCH_SIZE]
            rgb = torch.from_numpy(arrays.rgb[indices]).to(device, non_blocking=True)
            global_spectrum = torch.from_numpy(arrays.global_spectrum[indices]).to(device, non_blocking=True)
            local_tokens = torch.from_numpy(arrays.local_tokens[indices]).to(device, non_blocking=True)
            logits = _forward(state.model, candidate, rgb, global_spectrum, local_tokens, return_aux=False)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for array_index, probability in zip(indices, probabilities):
                sample_id = arrays.sample_ids[int(array_index)]
                if sample_id in set(state.fold.forbidden_outer_test_ids):
                    raise RuntimeError("outer-test sample reached inner prediction")
                rows.append({
                    "sample_id": sample_id,
                    "group_id": arrays.group_by_sample[sample_id],
                    "outer_fold_id": state.fold.outer_fold_id,
                    "fold_id": state.fold.fold_id,
                    "seed": SEED,
                    "inference_branch": candidate.inference_branch,
                    "y_true": int(arrays.labels[int(array_index)]),
                    "probability": float(probability),
                    "y_pred_fixed_0_5": int(float(probability) >= PRIMARY_THRESHOLD),
                })
    return rows


def _train_candidate_to_rung(execution_root: Path, candidate: ScreeningCandidate, folds: Sequence[ScreeningFold], arrays: HistologyArrays, id_to_index: Mapping[str, int], target_epoch: int, template: Any, environment: Any, registry: Any, device: torch.device, overall_started_ns: int) -> dict[str, Any]:
    candidate_root = execution_root / "candidates" / candidate.safe_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(candidate_root / "candidate.json", {**asdict(candidate), "config_hash": candidate.config_hash})
    states, pool_changes = _make_states(candidate, folds, registry, device)
    completed_epoch, history, best_score, selected_epoch = _load_resume(candidate_root, states, candidate, device)
    selected_path = candidate_root / "predictions" / "selected-exact-inner-oof.json"
    best_rows: list[dict[str, Any]] | None = None
    if selected_path.exists():
        best_rows = json.loads(selected_path.read_text(encoding="utf-8"))["records"]
    candidate_started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    for epoch in range(completed_epoch + 1, target_epoch + 1):
        fold_losses: dict[str, float] = {}
        for state in states:
            fold_losses[state.fold.fold_id] = _train_one_epoch(state, candidate, arrays, id_to_index, epoch, device)
            payload = _checkpoint_payload(state, candidate, epoch)
            _atomic_torch(candidate_root / "folds" / state.fold.fold_id / "recovery" / f"slot-{epoch % 2}.pt", payload)
        epoch_rows = [row for state in states for row in _predict(state, candidate, arrays, id_to_index, device)]
        metrics = compute_screening_metrics(epoch_rows)
        score = _score(metrics)
        history.append({
            "epoch": epoch,
            "score": score,
            "outer_fold_macro": metrics["outer_fold_macro"],
            "cross_context_pooled_oof": metrics["cross_context_pooled_oof"],
            "fold_losses": fold_losses,
            "completed_training_steps": max(state.completed_steps for state in states),
        })
        if score > best_score:
            best_score = score
            selected_epoch = epoch
            best_rows = epoch_rows
            for state in states:
                _atomic_torch(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "selected.pt", _checkpoint_payload(state, candidate, epoch))
            _atomic_json(selected_path, {
                "schema_version": "1.0",
                "candidate_id": candidate.candidate_id,
                "selected_epoch": epoch,
                "records": best_rows,
            })
        _atomic_json(candidate_root / "common-recovery.json", {
            "schema_version": "1.0",
            "candidate_id": candidate.candidate_id,
            "candidate_config_hash": candidate.config_hash,
            "completed_epoch": epoch,
            "best_score": best_score,
            "selected_epoch": selected_epoch,
            "history": history,
        })
        elapsed_ms = (time.monotonic_ns() - overall_started_ns) // 1_000_000
        measurements = RuntimeMeasurements(
            completed_epochs=epoch,
            completed_training_steps=max(state.completed_steps for state in states),
            wall_clock_duration_ms=int(elapsed_ms),
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
            device_identity=environment.device_identity,
            numeric_precision="float32",
            parameter_count=sum(parameter.numel() for parameter in states[0].model.parameters()),
            input_volume_bytes=arrays.input_volume_bytes,
        )
        assessment = assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch)
        if assessment.hard_exceeded:
            raise RuntimeError(f"hard screening budget exceeded at epoch {epoch}: {assessment.model_dump(mode='json')}")
        print(json.dumps({
            "stage": "training",
            "candidate_id": candidate.candidate_id,
            "epoch": epoch,
            "target_epoch": target_epoch,
            "outer_fold_macro_balanced_accuracy": score,
        }), flush=True)
    if best_rows is None or selected_epoch <= 0:
        raise RuntimeError("candidate did not produce selected inner OOF predictions")
    for state in states:
        _atomic_torch(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "last.pt", _checkpoint_payload(state, candidate, target_epoch))
    metrics = compute_screening_metrics(best_rows)
    _atomic_json(candidate_root / "predictions" / f"rung-{target_epoch:03d}-exact-inner-oof.json", {
        "schema_version": "1.0",
        "candidate_id": candidate.candidate_id,
        "selected_epoch": selected_epoch,
        "records": best_rows,
        "content_hash": hash_object(best_rows),
    })
    result = {
        "candidate_id": candidate.candidate_id,
        "safe_id": candidate.safe_id,
        "candidate_config_hash": candidate.config_hash,
        "model_key": candidate.model_key,
        "inference_branch": candidate.inference_branch,
        "role": candidate.role,
        "representation_key": candidate.representation_key,
        "training_recipe_key": candidate.training_recipe_key,
        "loss_key": candidate.loss_key,
        "alignment_weight": candidate.alignment_weight,
        "nce_weight": candidate.nce_weight,
        "required_control_keys": candidate.required_control_keys,
        "rung_epoch": target_epoch,
        "selected_epoch": selected_epoch,
        "score": _score(metrics),
        "metrics": metrics,
        "source_prediction_hash": hash_object(best_rows),
        "prediction_artifact": f"candidates/{candidate.safe_id}/predictions/rung-{target_epoch:03d}-exact-inner-oof.json",
        "completed_training_steps_per_fold": {state.fold.fold_id: state.completed_steps for state in states},
        "runtime_ms_this_invocation": int((time.monotonic_ns() - candidate_started_ns) // 1_000_000),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "parameter_count": sum(parameter.numel() for parameter in states[0].model.parameters()),
        "input_volume_bytes": arrays.input_volume_bytes,
        "deterministic_pooling_substitutions": pool_changes,
        "selected_checkpoints": {state.fold.fold_id: hash_file(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "selected.pt") for state in states},
        "last_checkpoints": {state.fold.fold_id: hash_file(candidate_root / "folds" / state.fold.fold_id / "checkpoints" / "last.pt") for state in states},
        "recovery_checkpoints": {state.fold.fold_id: hash_file(candidate_root / "folds" / state.fold.fold_id / "recovery" / f"slot-{target_epoch % 2}.pt") for state in states},
        "status": "complete",
    }
    _atomic_json(candidate_root / f"rung-{target_epoch:03d}-result.json", result)
    del states
    torch.cuda.empty_cache()
    return result


def _negative_record(execution_id: str, candidate: ScreeningCandidate, result: Mapping[str, Any], reason: str, details: Mapping[str, Any]) -> dict[str, Any]:
    nonce = hashlib.sha256(_json_bytes({"execution_id": execution_id, "candidate": candidate.candidate_id, "reason": reason, "details": details})).hexdigest()[:16]
    return {
        "schema_version": "1.0",
        "evidence_id": f"{execution_id}-{candidate.safe_id}-{reason}-{nonce}",
        "task_id": "15.12",
        "reason": reason,
        "candidate_id": candidate.candidate_id,
        "tested_configuration": asdict(candidate),
        "candidate_metrics": result.get("metrics", {}),
        "details": dict(details),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _append_negative(execution_root: Path, record: Mapping[str, Any]) -> None:
    _append_jsonl(execution_root / "negative-evidence.jsonl", record)
    _append_jsonl(OUTPUT_ROOT / "negative-evidence.jsonl", record)


def _rank(results: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(results, key=lambda item: (-float(item["score"]), str(item["candidate_id"]).encode("utf-8")))


def _fit_winner_thresholds(winner: ScreeningCandidate, winner_result: Mapping[str, Any], protocol_folds: Sequence[OuterFold], dataset_hash: str, protocol_hash: str, execution_root: Path) -> dict[str, Any]:
    selected = json.loads((execution_root / "candidates" / winner.safe_id / "predictions" / "selected-exact-inner-oof.json").read_text(encoding="utf-8"))["records"]
    wiring = HistologyStatisticsWiring()
    fitted: dict[str, Any] = {}
    for outer in protocol_folds:
        rows = [row for row in selected if row["outer_fold_id"] == outer.fold_id]
        normalized_rows = [{**row, "fold_id": f"{outer.fold_id}.inner-oof"} for row in rows]
        artifact = _artifact_from_rows(normalized_rows, artifact_id=f"{winner.safe_id}-{outer.fold_id}-inner-oof", run_id=f"{winner.safe_id}-{outer.fold_id}")
        values = wiring.fit_inner_oof(outer, artifact, dataset_hash=dataset_hash, protocol_hash=protocol_hash)
        threshold = float(values.threshold.state["threshold"])
        fitted_metrics = _metric_values(rows, label=f"{outer.fold_id}-fixed-0.5")
        threshold_artifact = _artifact_from_rows(normalized_rows, artifact_id=f"{winner.safe_id}-{outer.fold_id}-threshold", run_id=f"{winner.safe_id}-{outer.fold_id}")
        threshold_metrics = binary_metrics(threshold_artifact, _metric_schema(threshold)).model_dump(mode="json")
        fitted[outer.fold_id] = {
            "primary": {"label": "primary.fixed-threshold-0.5", "threshold": PRIMARY_THRESHOLD, "metrics": fitted_metrics},
            "fitted_threshold": {"label": "secondary.inner-oof-fitted-threshold", "threshold": threshold, "artifact": values.threshold.model_dump(mode="json"), "metrics": threshold_metrics},
            "fitted_temperature": {"label": "secondary.inner-oof-fitted-temperature", "temperature": float(values.temperature.state["temperature"]), "artifact": values.temperature.model_dump(mode="json")},
            "source_prediction_hash": artifact.content_hash,
            "source_sample_ids": outer.outer_train_ids,
            "forbidden_outer_test_ids": outer.outer_test_ids,
        }
    return fitted


def execute_screening() -> dict[str, Any]:
    for path in (WORKSPACE, DATA_ROOT, MANIFEST_PATH, TEMPLATE_PATH, OUTPUT_ROOT):
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError(f"task 15.12 is restricted to approved G: roots: {path}")
    template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.SCREENING:
        raise RuntimeError("task 15.12 requires the immutable screening template")
    environment = validate_live_environment(template)
    _set_determinism()
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    overall_started_ns = time.monotonic_ns()
    execution_id = datetime.now(timezone.utc).strftime("task-15-12-%Y%m%dT%H%M%S%fZ")
    execution_root = OUTPUT_ROOT / "executions" / execution_id
    execution_root.mkdir(parents=True, exist_ok=False)
    store = FailureAtomicArtifactStore(execution_root / "protocol-store")

    protected_roots = {"src": WORKSPACE / "src", "legacy_runs": WORKSPACE / "runs", "dataset_and_caches": DATA_ROOT}
    readonly_before = {name: _tree_fingerprint(path) for name, path in protected_roots.items()}
    manifest = _load_manifest()
    planner = HistologyNested5Planner(ProtocolPlanner(store))
    published_strict_protocol = planner.plan_and_publish(
        manifest,
        seed_set=(SEED,),
        relative_path="strict-nested-protocol.canonical.json",
    )
    strict_protocol = published_strict_protocol.manifest
    arrays = _load_arrays()
    if arrays.sample_ids != tuple(manifest.canonical_order):
        raise RuntimeError("cache row order differs from frozen Sample_Manifest Canonical_Order")
    group_by_sample = {record.sample_id: str(record.group_id) for record in manifest.records}
    if dict(arrays.group_by_sample) != group_by_sample:
        raise RuntimeError("cache patient mapping differs from frozen Sample_Manifest")
    folds = build_screening_folds(strict_protocol.folds, group_by_sample)
    id_to_index = {sample_id: index for index, sample_id in enumerate(arrays.sample_ids)}
    candidates = build_screening_candidates()

    fold_payload = [{
        **asdict(fold),
        "train_group_ids": tuple(sorted({group_by_sample[item] for item in fold.train_ids}, key=str.encode)),
    } for fold in folds]
    protocol_payload = {
        "schema_version": "1.0",
        "protocol_key": PROTOCOL_KEY,
        "strict_protocol_key": strict_protocol.protocol_key,
        "strict_protocol_hash": strict_protocol.protocol_hash,
        "dataset_hash": manifest.dataset_hash,
        "seed_set": [SEED],
        "folds": fold_payload,
        "selection": {
            "objective": "outer_fold_macro.inner_oof.balanced_accuracy",
            "direction": "maximize",
            "candidate_tie_break": "candidate_id_utf8_ascending",
            "epoch_tie_break": "earliest_epoch",
            "source": "25 patient-isolated inner folds only",
        },
        "outer_evaluation_budget": {"maximum_for_task": 0, "consumed": 0},
        "lopo_evaluation_budget": {"maximum_for_task": 0, "consumed": 0},
    }
    protocol_hash = hash_object(protocol_payload)
    metric_configuration = _metric_schema().model_dump(mode="json")
    plan = {
        "schema_version": "1.0",
        "task_id": "15.12",
        "execution_id": execution_id,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_hash": manifest.dataset_hash,
        "protocol": {**protocol_payload, "protocol_hash": protocol_hash},
        "fold_manifest_hash": hash_object(fold_payload),
        "metric_configuration": metric_configuration,
        "metric_configuration_hash": hash_object(metric_configuration),
        "candidate_set": [{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates],
        "candidate_set_hash": hash_object([asdict(candidate) for candidate in candidates]),
        "successive_halving": {
            "fully_predeclared": True,
            "rung_epochs": list(RUNG_EPOCHS),
            "survivors": list(RUNG_SURVIVORS),
            "all_candidates_receive_identical_initial_budget": True,
            "inner_folds_per_candidate": len(folds),
            "ranking": "outer-fold macro inner-OOF balanced accuracy then candidate ID",
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "augmentation": "none",
            "sampling": "sample-level deterministic epoch permutation",
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
            "resource_budget_hash": hash_object(template.resource_budget.model_dump(mode="json")),
        },
        "scope": {
            "outer_labels_allowed": False,
            "outer_evaluation_allowed": False,
            "lopo_allowed": False,
            "write_root": "runs_v2/screening/histology",
            "read_only_roots": ["src", "runs", "dataset", "caches"],
        },
    }
    _atomic_json(execution_root / "frozen-screening-plan.json", plan)
    _atomic_json(execution_root / "environment-lock.json", environment.model_dump(mode="json"))
    _atomic_json(execution_root / "protocol.json", {**protocol_payload, "protocol_hash": protocol_hash})

    # Model construction occurs only after both the strict nested protocol and
    # the screening protocol/plan have been atomically published.
    registry = create_histology_model_registry()
    for candidate in candidates:
        registration = registry.resolve(candidate.model_key)
        if registration.input_contract.representation_key != candidate.representation_key:
            raise RuntimeError(f"Representation_Key mismatch for {candidate.candidate_id}")

    active = list(candidates)
    all_results: dict[str, dict[str, Any]] = {}
    rung_tables: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rung_epoch, survivor_count in zip(RUNG_EPOCHS, RUNG_SURVIVORS):
        rung_results: list[dict[str, Any]] = []
        for candidate in active:
            try:
                result = _train_candidate_to_rung(execution_root, candidate, folds, arrays, id_to_index, rung_epoch, template, environment, registry, device, overall_started_ns)
                all_results[candidate.candidate_id] = result
                rung_results.append(result)
            except Exception as exc:
                failure = {"candidate_id": candidate.candidate_id, "rung_epoch": rung_epoch, "error": repr(exc)}
                failures.append(failure)
                _append_negative(execution_root, _negative_record(execution_id, candidate, {}, "failed_or_unstable", failure))
                print(json.dumps({"stage": "candidate_failed", **failure}), file=sys.stderr, flush=True)
        ranked = _rank(rung_results)
        if not ranked:
            raise RuntimeError(f"all candidates failed at rung {rung_epoch}")
        promoted_ids = {str(item["candidate_id"]) for item in ranked[:survivor_count]}
        for rank, item in enumerate(ranked[survivor_count:], start=survivor_count + 1):
            candidate = next(value for value in active if value.candidate_id == item["candidate_id"])
            _append_negative(execution_root, _negative_record(execution_id, candidate, item, "underperformance", {
                "rung_epoch": rung_epoch,
                "rank": rank,
                "promoted_count": survivor_count,
                "best_promoted_score": float(ranked[0]["score"]),
                "candidate_score": float(item["score"]),
                "deterministic_pruning_rule": "score descending then candidate_id UTF-8 ascending",
            }))
        table = {
            "rung_epoch": rung_epoch,
            "survivor_count": survivor_count,
            "ranked_candidates": [{
                "rank": rank,
                "candidate_id": item["candidate_id"],
                "role": item["role"],
                "representation_key": item["representation_key"],
                "selected_epoch": item["selected_epoch"],
                "score": item["score"],
                "outer_fold_macro": item["metrics"]["outer_fold_macro"],
                "cross_context_pooled_oof": item["metrics"]["cross_context_pooled_oof"],
                "promoted": item["candidate_id"] in promoted_ids,
            } for rank, item in enumerate(ranked, start=1)],
        }
        rung_tables.append(table)
        _atomic_json(execution_root / f"rung-{rung_epoch:03d}-candidate-table.json", table)
        active = [candidate for candidate in active if candidate.candidate_id in promoted_ids]

    final_ranked = _rank(all_results[candidate.candidate_id] for candidate in active)
    winner_result = dict(final_ranked[0])
    winner = next(candidate for candidate in candidates if candidate.candidate_id == winner_result["candidate_id"])
    threshold_calibration = _fit_winner_thresholds(winner, winner_result, strict_protocol.folds, manifest.dataset_hash, protocol_hash, execution_root)
    candidate_scores = {key: value["score"] for key, value in sorted(all_results.items(), key=lambda item: item[0].encode("utf-8"))}
    local = next(candidate for candidate in candidates if candidate.candidate_id == LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY)
    initial_scores = {item["candidate_id"]: float(item["score"]) for item in rung_tables[0]["ranked_candidates"]}
    matched_control_comparisons = []
    for control_key in local.required_control_keys:
        control = all_results[control_key]
        matched_control_comparisons.append({
            "candidate_id": local.candidate_id,
            "control_id": control_key,
            "comparison_rung_epoch": RUNG_EPOCHS[0],
            "same_dataset_hash": True,
            "same_protocol_hash": True,
            "same_seed_set": True,
            "same_fold_manifest_hash": True,
            "same_metric_configuration_hash": True,
            "same_resource_budget_hash": True,
            "same_attained_budget": True,
            "same_representation_key": local.representation_key == control["representation_key"],
            "balanced_accuracy_difference": initial_scores[local.candidate_id] - initial_scores[control_key],
            "candidate_score": initial_scores[local.candidate_id],
            "control_score": initial_scores[control_key],
        })
    frozen_selection = {
        "schema_version": "1.0",
        "decision_id": f"{execution_id}-histology-selection",
        "candidate_set": [{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates],
        "candidate_scores": candidate_scores,
        "candidate_status": {candidate.candidate_id: (
            "winner" if candidate.candidate_id == winner.candidate_id else
            "completed_dominated_or_pruned" if candidate.candidate_id in all_results else
            "failed_or_unstable"
        ) for candidate in candidates},
        "metric_direction": "maximize",
        "objective_metric": "outer_fold_macro.inner_oof.balanced_accuracy",
        "tie_break_rule": "candidate_id_utf8_ascending",
        "source_sample_ids_by_outer_fold": {outer.fold_id: outer.outer_train_ids for outer in strict_protocol.folds},
        "forbidden_sample_ids_by_outer_fold": {outer.fold_id: outer.outer_test_ids for outer in strict_protocol.folds},
        "source_prediction_hash": winner_result["source_prediction_hash"],
        "selected_candidate": winner.candidate_id,
        "frozen_configuration": {
            **asdict(winner),
            "selected_epoch": winner_result["selected_epoch"],
            "epoch_policy": "earliest maximum outer-fold-macro inner-OOF balanced accuracy within attained budget",
            "primary_threshold": PRIMARY_THRESHOLD,
            "threshold_calibration_by_outer_fold": threshold_calibration,
            "preprocessing": {
                "pca_only": "cached spatial-mean PCA bytes, read-only, distinct Representation_Key",
                "rgb_only": "unit-interval RGB cache",
                "local_models": "cached 4x4 local PCA tokens, read-only, distinct Representation_Key",
            },
            "seed": SEED,
            "required_controls_for_strict_attribution": [candidate.candidate_id for candidate in candidates if candidate.candidate_id != winner.candidate_id],
        },
        "selection_evidence": "nested inner-patient OOF only; no strict outer or LOPO prediction",
        "outer_evaluation_budget_consumed": 0,
        "lopo_evaluation_budget_consumed": 0,
        "frozen_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    frozen_selection["decision_hash"] = hash_object(frozen_selection)
    _atomic_json(execution_root / "frozen-selection-decision.json", frozen_selection)
    _atomic_json(execution_root / "matched-control-comparisons.json", matched_control_comparisons)

    readonly_after = {name: _tree_fingerprint(path) for name, path in protected_roots.items()}
    if readonly_before != readonly_after:
        raise RuntimeError("read-only src, legacy runs, dataset, or cache metadata changed")
    elapsed_ms = int((time.monotonic_ns() - overall_started_ns) // 1_000_000)
    report = {
        "schema_version": "1.0",
        "task_id": "15.12",
        "status": "pass" if not failures else "completed_with_negative_evidence",
        "execution_id": execution_id,
        "command": "conda run --no-capture-output -n TCFM python -m benchmark.runners.histology_screening --execute",
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "dataset_key": "histology-hsi-gb",
            "evidence_scope": "nested inner-patient OOF screening only",
            "strict_outer_predictions_created": False,
            "strict_outer_labels_read": False,
            "strict_outer_evaluation_budget_consumed": 0,
            "lopo_started": False,
            "lopo_evaluation_budget_consumed": 0,
            "strict_multiseed_started": False,
            "single_process": True,
            "read_only_inputs_unchanged": True,
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
            "manifest_hash": hash_file(MANIFEST_PATH),
            "protocol_key": PROTOCOL_KEY,
            "protocol_hash": protocol_hash,
            "strict_protocol_key": strict_protocol.protocol_key,
            "strict_protocol_hash": strict_protocol.protocol_hash,
            "fold_manifest_hash": plan["fold_manifest_hash"],
            "candidate_set_hash": plan["candidate_set_hash"],
            "metric_configuration_hash": plan["metric_configuration_hash"],
            "resource_budget_hash": plan["template"]["resource_budget_hash"],
            "template_hash": template.template_hash,
            "environment_lock_hash": environment.lock_hash,
            "runner_hash": hash_file(Path(__file__)),
            "model_source_hash": hash_file(WORKSPACE / "benchmark" / "models" / "histology.py"),
            "legacy_model_source_hash": hash_file(WORKSPACE / "src" / "hypercross_routeA.py"),
        },
        "budget": {
            "frozen_limits": template.resource_budget.model_dump(mode="json"),
            "successive_halving": plan["successive_halving"],
            "observed_wall_clock_duration_ms": elapsed_ms,
            "observed_peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
            "input_volume_bytes": arrays.input_volume_bytes,
        },
        "folds": [{
            "outer_fold_id": fold.outer_fold_id,
            "inner_fold_id": fold.fold_id,
            "train_count": len(fold.train_ids),
            "validation_count": len(fold.validation_ids),
            "validation_group_ids": fold.validation_group_ids,
            "forbidden_outer_test_count": len(fold.forbidden_outer_test_ids),
            "outer_test_excluded": not bool((set(fold.train_ids) | set(fold.validation_ids)) & set(fold.forbidden_outer_test_ids)),
        } for fold in folds],
        "rung_candidate_tables": rung_tables,
        "candidate_table": [{
            "candidate_id": key,
            "model_key": value["model_key"],
            "role": value["role"],
            "representation_key": value["representation_key"],
            "attained_epoch": value["rung_epoch"],
            "selected_epoch": value["selected_epoch"],
            "score": value["score"],
            "metrics": value["metrics"],
            "runtime_ms": value["runtime_ms_this_invocation"],
            "peak_device_memory_bytes": value["peak_device_memory_bytes"],
            "parameter_count": value["parameter_count"],
            "prediction_artifact": value["prediction_artifact"],
        } for key, value in sorted(all_results.items(), key=lambda item: item[0].encode("utf-8"))],
        "winner": winner_result,
        "threshold_calibration": threshold_calibration,
        "matched_control_comparisons": matched_control_comparisons,
        "frozen_selection": frozen_selection,
        "failures": failures,
        "negative_evidence_paths": ["negative-evidence.jsonl", "../../negative-evidence.jsonl"],
        "artifacts": {
            "frozen_plan": "frozen-screening-plan.json",
            "protocol": "protocol.json",
            "environment_lock": "environment-lock.json",
            "frozen_selection": "frozen-selection-decision.json",
            "matched_control_comparisons": "matched-control-comparisons.json",
            "candidate_root": "candidates/",
        },
        "caveats": [
            "This is single-seed screening evidence and is not strict multi-seed or LOPO evidence.",
            "No strict outer prediction was generated and no strict outer or LOPO evaluation budget was consumed.",
            "Successive halving was frozen before training; every candidate received all 25 inner folds through epoch 10 before deterministic pruning.",
            "The task-level winner is based only on macro aggregation of five outer-train inner-OOF scores; each contributing score excludes that outer fold's patient labels.",
            "Primary classification metrics retain threshold 0.5. Inner-OOF fitted thresholds and temperatures are separately labeled per outer fold.",
            "The cached spatial-mean PCA and cached local spatial PCA have distinct Representation_Keys. Cache bytes are hash-bound, but their historical creation lineage cannot prove fold-scoped PCA fitting, so this remains a strict-evaluation blocker unless regenerated from raw cubes.",
            "Single-seed screening cannot establish independent-seed stability.",
        ],
    }
    _atomic_json(execution_root / "task-15.12-report.json", report)
    summary = {
        "status": report["status"],
        "execution_id": execution_id,
        "report": (execution_root / "task-15.12-report.json").relative_to(WORKSPACE).as_posix(),
        "selected_candidate": winner.candidate_id,
        "selected_epoch": winner_result["selected_epoch"],
        "primary_balanced_accuracy": winner_result["score"],
        "strict_outer_evaluation_budget_consumed": 0,
        "lopo_evaluation_budget_consumed": 0,
    }
    _atomic_json(OUTPUT_ROOT / "latest-task-15.12-summary.json", summary)
    return summary


def finalize_existing(execution_id: str) -> dict[str, Any]:
    """Finalize a fully trained execution after a report-stage interruption."""

    execution_root = OUTPUT_ROOT / "executions" / execution_id
    if not execution_root.is_dir():
        raise FileNotFoundError(f"unknown screening execution: {execution_id}")
    report_path = execution_root / "task-15.12-report.json"
    if report_path.exists():
        raise RuntimeError("completed report already exists; append-only finalization refused")
    plan = json.loads((execution_root / "frozen-screening-plan.json").read_text(encoding="utf-8"))
    environment = json.loads((execution_root / "environment-lock.json").read_text(encoding="utf-8"))
    protocol_payload = json.loads((execution_root / "protocol.json").read_text(encoding="utf-8"))
    strict_protocol = ProtocolPlanner.reload((execution_root / "protocol-store" / "strict-nested-protocol.canonical.json").read_bytes())
    manifest = _load_manifest()
    candidates = build_screening_candidates()
    all_results: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_root = execution_root / "candidates" / candidate.safe_id
        result_paths = sorted(candidate_root.glob("rung-*-result.json"), key=lambda path: path.name.encode("utf-8"))
        if not result_paths:
            raise RuntimeError(f"candidate has no completed rung result: {candidate.candidate_id}")
        all_results[candidate.candidate_id] = json.loads(result_paths[-1].read_text(encoding="utf-8"))
    rung_tables = [json.loads((execution_root / f"rung-{epoch:03d}-candidate-table.json").read_text(encoding="utf-8")) for epoch in RUNG_EPOCHS]
    final_table = rung_tables[-1]["ranked_candidates"]
    if len(final_table) != 1:
        raise RuntimeError("final rung must contain exactly one winner")
    winner = next(candidate for candidate in candidates if candidate.candidate_id == final_table[0]["candidate_id"])
    winner_result = all_results[winner.candidate_id]
    threshold_calibration = _fit_winner_thresholds(winner, winner_result, strict_protocol.folds, manifest.dataset_hash, protocol_payload["protocol_hash"], execution_root)

    local = next(candidate for candidate in candidates if candidate.candidate_id == LOCAL_ROUTE_A_ALIGNMENT_NCE_CANDIDATE_KEY)
    initial_scores = {item["candidate_id"]: float(item["score"]) for item in rung_tables[0]["ranked_candidates"]}
    matched_control_comparisons = []
    for control_key in local.required_control_keys:
        control = all_results[control_key]
        matched_control_comparisons.append({
            "candidate_id": local.candidate_id,
            "control_id": control_key,
            "comparison_rung_epoch": RUNG_EPOCHS[0],
            "same_dataset_hash": True,
            "same_protocol_hash": True,
            "same_seed_set": True,
            "same_fold_manifest_hash": True,
            "same_metric_configuration_hash": True,
            "same_resource_budget_hash": True,
            "same_attained_budget": True,
            "same_representation_key": local.representation_key == control["representation_key"],
            "balanced_accuracy_difference": initial_scores[local.candidate_id] - initial_scores[control_key],
            "candidate_score": initial_scores[local.candidate_id],
            "control_score": initial_scores[control_key],
        })
    _atomic_json(execution_root / "matched-control-comparisons.json", matched_control_comparisons)

    candidate_scores = {key: value["score"] for key, value in sorted(all_results.items(), key=lambda item: item[0].encode("utf-8"))}
    required_controls_for_strict = [candidate.candidate_id for candidate in candidates if candidate.candidate_id != winner.candidate_id]
    frozen_selection = {
        "schema_version": "1.0",
        "decision_id": f"{execution_id}-histology-selection",
        "candidate_set": plan["candidate_set"],
        "candidate_scores": candidate_scores,
        "candidate_status": {candidate.candidate_id: ("winner" if candidate.candidate_id == winner.candidate_id else "completed_dominated_or_pruned") for candidate in candidates},
        "metric_direction": "maximize",
        "objective_metric": "outer_fold_macro.inner_oof.balanced_accuracy",
        "tie_break_rule": "candidate_id_utf8_ascending",
        "source_sample_ids_by_outer_fold": {outer.fold_id: outer.outer_train_ids for outer in strict_protocol.folds},
        "forbidden_sample_ids_by_outer_fold": {outer.fold_id: outer.outer_test_ids for outer in strict_protocol.folds},
        "source_prediction_hash": winner_result["source_prediction_hash"],
        "selected_candidate": winner.candidate_id,
        "frozen_configuration": {
            **asdict(winner),
            "selected_epoch": winner_result["selected_epoch"],
            "epoch_policy": "earliest maximum outer-fold-macro inner-OOF balanced accuracy within attained budget",
            "primary_threshold": PRIMARY_THRESHOLD,
            "threshold_calibration_by_outer_fold": threshold_calibration,
            "preprocessing": {
                "pca_only": "cached spatial-mean PCA bytes, read-only, distinct Representation_Key",
                "rgb_only": "unit-interval RGB cache",
                "local_models": "cached 4x4 local PCA tokens, read-only, distinct Representation_Key",
            },
            "seed": SEED,
            "required_controls_for_strict_attribution": required_controls_for_strict,
        },
        "selection_evidence": "nested inner-patient OOF only; no strict outer or LOPO prediction",
        "outer_evaluation_budget_consumed": 0,
        "lopo_evaluation_budget_consumed": 0,
        "recovered_from_report_stage_failure": True,
        "frozen_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    frozen_selection["decision_hash"] = hash_object(frozen_selection)
    _atomic_json(execution_root / "frozen-selection-decision.json", frozen_selection)

    failure_record = {
        "schema_version": "1.0",
        "evidence_id": f"{execution_id}-report-finalization-fold-identity",
        "task_id": "15.12",
        "reason": "report_finalization_failure_recovered",
        "candidate_id": winner.candidate_id,
        "tested_configuration": asdict(winner),
        "candidate_metrics": winner_result["metrics"],
        "details": {
            "failed_stage": "threshold_metric_artifact_assembly",
            "error": "derived aggregate artifact retained multiple inner fold IDs",
            "training_and_checkpoints_unchanged": True,
            "recovery_action": "normalize only the derived aggregate fold identity and finalize from immutable selected OOF predictions",
        },
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _append_negative(execution_root, failure_record)
    peak_memory = max(int(value["peak_device_memory_bytes"]) for value in all_results.values())
    started = datetime.fromisoformat(plan["created_utc"].replace("Z", "+00:00"))
    elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    report = {
        "schema_version": "1.0",
        "task_id": "15.12",
        "status": "completed_with_recovered_report_failure_and_negative_evidence",
        "execution_id": execution_id,
        "commands": [
            "conda run --no-capture-output -n TCFM python -m benchmark.runners.histology_screening --execute",
            f"conda run --no-capture-output -n TCFM python -m benchmark.runners.histology_screening --finalize-execution {execution_id}",
        ],
        "completed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "dataset_key": "histology-hsi-gb",
            "evidence_scope": "nested inner-patient OOF screening only",
            "strict_outer_predictions_created": False,
            "strict_outer_labels_read": False,
            "strict_outer_evaluation_budget_consumed": 0,
            "lopo_started": False,
            "lopo_evaluation_budget_consumed": 0,
            "strict_multiseed_started": False,
            "single_process": True,
            "write_root": "runs_v2/screening/histology",
        },
        "environment": environment,
        "identity": {
            "dataset_hash": manifest.dataset_hash,
            "manifest_hash": hash_file(MANIFEST_PATH),
            "protocol_key": protocol_payload["protocol_key"],
            "protocol_hash": protocol_payload["protocol_hash"],
            "strict_protocol_key": strict_protocol.protocol_key,
            "strict_protocol_hash": strict_protocol.protocol_hash,
            "fold_manifest_hash": plan["fold_manifest_hash"],
            "candidate_set_hash": plan["candidate_set_hash"],
            "metric_configuration_hash": plan["metric_configuration_hash"],
            "resource_budget_hash": plan["template"]["resource_budget_hash"],
            "template_hash": plan["template"]["template_hash"],
            "environment_lock_hash": environment["lock_hash"],
            "runner_hash_after_report_fix": hash_file(Path(__file__)),
            "model_source_hash": hash_file(WORKSPACE / "benchmark" / "models" / "histology.py"),
            "legacy_model_source_hash": hash_file(WORKSPACE / "src" / "hypercross_routeA.py"),
        },
        "budget": {
            "frozen_limits": plan["template"]["resource_budget"],
            "successive_halving": plan["successive_halving"],
            "observed_wall_clock_duration_ms_including_recovery": elapsed_ms,
            "observed_peak_device_memory_bytes": peak_memory,
            "input_volume_bytes": winner_result["input_volume_bytes"],
        },
        "folds": [{
            "outer_fold_id": fold["outer_fold_id"],
            "inner_fold_id": fold["fold_id"],
            "train_count": len(fold["train_ids"]),
            "validation_count": len(fold["validation_ids"]),
            "validation_group_ids": fold["validation_group_ids"],
            "forbidden_outer_test_count": len(fold["forbidden_outer_test_ids"]),
            "outer_test_excluded": True,
        } for fold in plan["protocol"]["folds"]],
        "rung_candidate_tables": rung_tables,
        "candidate_table": [{
            "candidate_id": key,
            "model_key": value["model_key"],
            "role": value["role"],
            "representation_key": value["representation_key"],
            "attained_epoch": value["rung_epoch"],
            "selected_epoch": value["selected_epoch"],
            "score": value["score"],
            "metrics": value["metrics"],
            "runtime_ms_last_rung": value["runtime_ms_this_invocation"],
            "peak_device_memory_bytes": value["peak_device_memory_bytes"],
            "parameter_count": value["parameter_count"],
            "prediction_artifact": value["prediction_artifact"],
            "selected_checkpoints": value["selected_checkpoints"],
            "last_checkpoints": value["last_checkpoints"],
            "recovery_checkpoints": value["recovery_checkpoints"],
        } for key, value in sorted(all_results.items(), key=lambda item: item[0].encode("utf-8"))],
        "winner": winner_result,
        "threshold_calibration": threshold_calibration,
        "matched_control_comparisons": matched_control_comparisons,
        "frozen_selection": frozen_selection,
        "failures": [failure_record],
        "negative_evidence_paths": ["negative-evidence.jsonl", "runs_v2/screening/histology/negative-evidence.jsonl"],
        "artifacts": {
            "frozen_plan": "frozen-screening-plan.json",
            "protocol": "protocol.json",
            "strict_protocol": "protocol-store/strict-nested-protocol.canonical.json",
            "environment_lock": "environment-lock.json",
            "frozen_selection": "frozen-selection-decision.json",
            "matched_control_comparisons": "matched-control-comparisons.json",
            "candidate_root": "candidates/",
        },
        "caveats": [
            "This is single-seed screening evidence and is not strict multi-seed or LOPO evidence.",
            "No strict outer prediction was generated and no strict outer or LOPO evaluation budget was consumed.",
            "Every candidate received all 25 inner folds through epoch 10 before deterministic pruning.",
            "The task-level winner uses only macro aggregation of five outer-train inner-OOF scores; each contributing score excludes that outer fold's patient labels.",
            "Primary metrics retain threshold 0.5; fitted thresholds and temperatures are separately labeled per outer fold.",
            "Cached spatial-mean PCA and cached local spatial PCA use distinct Representation_Keys, but historical cache creation lineage cannot prove fold-scoped PCA fitting. Regeneration from raw cubes remains a strict-evaluation blocker.",
            "Single-seed screening cannot establish independent-seed stability.",
            "The report-stage fold-identity failure occurred after all training and checkpoint publication; it was appended as negative evidence and recovered without changing training outputs.",
        ],
    }
    _atomic_json(report_path, report)
    summary = {
        "status": report["status"],
        "execution_id": execution_id,
        "report": report_path.relative_to(WORKSPACE).as_posix(),
        "selected_candidate": winner.candidate_id,
        "selected_epoch": winner_result["selected_epoch"],
        "primary_balanced_accuracy": winner_result["score"],
        "strict_outer_evaluation_budget_consumed": 0,
        "lopo_evaluation_budget_consumed": 0,
    }
    _atomic_json(OUTPUT_ROOT / "latest-task-15.12-summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true", help="run complete governed task-15.12 screening")
    group.add_argument("--finalize-execution", metavar="EXECUTION_ID", help="finalize a trained execution after report-stage interruption")
    arguments = parser.parse_args(argv)
    try:
        result = finalize_existing(arguments.finalize_execution) if arguments.finalize_execution else execute_screening()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
