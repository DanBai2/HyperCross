"""Task 15.13 cache-bound strict Histology multi-seed nested CV and LOPO.

The runner deliberately does not reuse task-15.12's cross-context winner as a
nested-CV decision.  It trains the frozen candidate/control set on each outer
fold's five inner-patient folds, freezes one candidate/epoch decision per outer
fold, and only then evaluates that outer fold.  The fixed LOPO candidate and
epoch are derived from those nested inner decisions before any LOPO label is
read.

The historical local-token and spatial-mean caches are content-hash bound but
their creation lineage is incomplete.  Consequently this executor emits a
complete cache-bound strict-protocol track whose claim is capped at
``strict_candidate``.  It never calls the cache bytes full raw-lineage evidence.
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
from torch import Tensor, nn

from benchmark.cli_runtime import validate_live_environment
from benchmark.fast_lane import FastLaneStage, load_fast_lane_template
from benchmark.metrics import binary_metrics, paired_patient_compare, patient_bootstrap
from benchmark.models.histology import create_histology_model_registry
from benchmark.protocols import ProtocolPlanner
from benchmark.protocols.histology_lopo_fixed import HistologyFixedEpochPolicy, HistologyLopoFixedPlanner
from benchmark.protocols.histology_nested import HistologyNested5Planner
from benchmark.recovery import FailureAtomicArtifactStore
from benchmark.reporting import ClaimEvidence, ClaimGate
from benchmark.runners.histology_baseline import HistologyArrays, _load_arrays, _tree_fingerprint
from benchmark.runners.histology_screening import (
    BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    PRIMARY_THRESHOLD,
    WEIGHT_DECAY,
    ScreeningCandidate,
    ScreeningFold,
    _forward,
    _loss,
    _replace_deterministic_pools,
    build_screening_candidates,
    build_screening_folds,
)
from benchmark.schema import (
    FittedArtifact,
    ImplementationKind,
    MetricSchema,
    PredictionArtifact,
    PredictionKey,
    PredictionRecord,
    RuntimeMeasurements,
    SampleManifest,
    TaskKind,
    assess_resource_budget,
    canonical_serialize,
    hash_file,
    hash_object,
)
from benchmark.transforms import PreprocessingManager


WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\PKG - HistologyHSI-GB")
MANIFEST_PATH = WORKSPACE / "artifacts" / "manifests" / "histology-hsi-gb" / "ba6a973ffa80636d42d50e274719bb72b4907766056b63e90d078da472704e12.json"
TEMPLATE_PATH = WORKSPACE / "configs" / "fast_lane" / "strict.json"
SCREENING_SUMMARY = WORKSPACE / "runs_v2" / "screening" / "histology" / "latest-task-15.12-summary.json"
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "strict" / "histology"
STRICT_SEEDS = (7, 42, 123)
INNER_SELECTION_SEED = 42
INNER_SELECTION_MAX_EPOCHS = 40
BOOTSTRAP_SEED = 2027
RAW_CUBE_TOTAL_BYTES = 622_309_721_600
COMMAND_EXECUTE = "conda run --no-capture-output -n TCFM python -m benchmark.runners.histology_strict --execute"


@dataclass
class TrainState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    completed_steps: int = 0


@dataclass(frozen=True)
class EvalFold:
    fold_id: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _publish_json_once(path: Path, value: Any) -> None:
    content = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    _atomic_bytes(path, content)


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


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return result


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


def _load_manifest() -> SampleManifest:
    manifest = SampleManifest.model_validate(_read_json(MANIFEST_PATH))
    if manifest.dataset_hash != MANIFEST_PATH.stem or len(manifest.records) != 469:
        raise RuntimeError("frozen Histology manifest identity is invalid")
    return manifest


def _metric_schema() -> MetricSchema:
    return MetricSchema(task_kind=TaskKind.BINARY_CLASSIFICATION, numeric_tolerance=1e-12, decision_threshold=0.5, ece_bins=10)


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _candidate_map() -> dict[str, ScreeningCandidate]:
    return {candidate.candidate_id: candidate for candidate in build_screening_candidates()}


def validate_screening_checkpoint(summary: Mapping[str, Any], decision: Mapping[str, Any]) -> tuple[ScreeningCandidate, ...]:
    """Validate task-15.12 identity without adopting its global winner decision."""

    if not str(summary.get("status", "")).startswith("completed"):
        raise RuntimeError("task-15.12 screening checkpoint is not complete")
    expected_hash = str(decision.get("decision_hash", ""))
    unhashed = dict(decision)
    unhashed.pop("decision_hash", None)
    if hash_object(unhashed) != expected_hash:
        raise RuntimeError("task-15.12 frozen decision hash mismatch")
    declared = _candidate_map()
    frozen = decision.get("candidate_set")
    if not isinstance(frozen, list) or len(frozen) != len(declared):
        raise RuntimeError("task-15.12 candidate set is incomplete")
    for item in frozen:
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in declared or item.get("config_hash") != declared[candidate_id].config_hash:
            raise RuntimeError(f"task-15.12 candidate identity mismatch: {candidate_id}")
    frozen_configuration = decision.get("frozen_configuration", {})
    required = set(str(value) for value in frozen_configuration.get("required_controls_for_strict_attribution", ()))
    required.add(str(decision.get("selected_candidate")))
    missing = sorted(required - set(declared), key=str.encode)
    if missing:
        raise RuntimeError("required strict controls are missing: " + ", ".join(missing))
    # All eight governed candidates/controls are retained because per-outer-fold
    # selection must not inherit the cross-context task-15.12 winner.
    return tuple(declared[key] for key in sorted(declared, key=str.encode))


def choose_outer_decisions(candidate_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Choose candidate/epoch independently for each outer fold.

    ``candidate_evidence[candidate_id][outer_fold_id]`` must contain only that
    outer fold's five-inner-patient OOF score, epoch, source IDs, and row hash.
    """

    outer_ids = sorted({outer for evidence in candidate_evidence.values() for outer in evidence}, key=str.encode)
    decisions: dict[str, dict[str, Any]] = {}
    for outer_id in outer_ids:
        options = []
        for candidate_id, by_outer in candidate_evidence.items():
            if outer_id not in by_outer:
                raise RuntimeError(f"candidate {candidate_id} lacks inner evidence for {outer_id}")
            item = by_outer[outer_id]
            options.append((
                -float(item["score"]),
                int(item["epoch"]),
                candidate_id.encode("utf-8"),
                candidate_id,
                item,
            ))
        _, _, _, winner_id, winner = min(options)
        decisions[outer_id] = {
            "selected_candidate": winner_id,
            "selected_epoch": int(winner["epoch"]),
            "selected_score": float(winner["score"]),
            "source_prediction_hash": str(winner["source_prediction_hash"]),
            "source_sample_ids": tuple(winner["source_sample_ids"]),
            "candidate_scores": {candidate_id: float(by_outer[outer_id]["score"]) for candidate_id, by_outer in sorted(candidate_evidence.items())},
            "candidate_epochs": {candidate_id: int(by_outer[outer_id]["epoch"]) for candidate_id, by_outer in sorted(candidate_evidence.items())},
            "metric_direction": "maximize",
            "tie_break_rule": "score descending, epoch ascending, candidate_id UTF-8 ascending",
        }
    return decisions


def freeze_lopo_policy(outer_decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze LOPO candidate and epoch from nested inner evidence only."""

    if not outer_decisions:
        raise ValueError("outer decisions cannot be empty")
    candidate_scores: dict[str, list[float]] = {}
    for decision in outer_decisions.values():
        for candidate_id, score in decision["candidate_scores"].items():
            candidate_scores.setdefault(candidate_id, []).append(float(score))
    ranked = sorted(
        ((-float(np.mean(scores)), candidate_id.encode("utf-8"), candidate_id) for candidate_id, scores in candidate_scores.items()),
    )
    selected_candidate = ranked[0][2]
    selected_outer_epochs = sorted(int(value["selected_epoch"]) for value in outer_decisions.values())
    fixed_epoch = selected_outer_epochs[len(selected_outer_epochs) // 2]
    payload = {
        "selection_source": "nested per-outer-fold inner-patient OOF only",
        "outer_labels_used": False,
        "selected_candidate": selected_candidate,
        "fixed_epoch": fixed_epoch,
        "candidate_mean_inner_scores": {key: float(np.mean(values)) for key, values in sorted(candidate_scores.items())},
        "nested_selected_epochs": selected_outer_epochs,
        "epoch_rule": "median of per-outer-fold selected epochs; odd fold count",
        "candidate_rule": "highest mean of five outer-specific inner-OOF scores; candidate ID tie-break",
    }
    payload["policy_hash"] = hash_object(payload)
    return payload


def _state(candidate: ScreeningCandidate, seed: int, fold_ordinal: int, max_epochs: int, registry: Any, device: torch.device) -> TrainState:
    initialization_seed = seed * 1_000_003 + fold_ordinal * 10_007
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(initialization_seed)
        torch.cuda.manual_seed_all(initialization_seed)
        model = registry.build(candidate.model_key)
    _replace_deterministic_pools(model, candidate)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return TrainState(model=model, optimizer=optimizer, scheduler=scheduler)


def _indices(ids: Iterable[str], id_to_index: Mapping[str, int]) -> np.ndarray:
    return np.asarray([id_to_index[item] for item in ids], dtype=np.int64)


def _train_epoch(state: TrainState, candidate: ScreeningCandidate, train_ids: Sequence[str], arrays: HistologyArrays, id_to_index: Mapping[str, int], seed: int, fold_ordinal: int, epoch: int, device: torch.device) -> float:
    train_indices = _indices(train_ids, id_to_index)
    order = np.random.default_rng(seed * 1_000_003 + fold_ordinal * 10_007 + epoch).permutation(train_indices)
    state.model.train()
    loss_sum = 0.0
    seen = 0
    for start in range(0, len(order), BATCH_SIZE):
        selected = order[start:start + BATCH_SIZE]
        rgb = torch.from_numpy(arrays.rgb[selected]).to(device, non_blocking=True)
        spectrum = torch.from_numpy(arrays.global_spectrum[selected]).to(device, non_blocking=True)
        local = torch.from_numpy(arrays.local_tokens[selected]).to(device, non_blocking=True)
        labels = torch.from_numpy(arrays.labels[selected]).to(device, non_blocking=True)
        state.optimizer.zero_grad(set_to_none=True)
        needs_aux = candidate.alignment_weight > 0 or candidate.nce_weight > 0
        output = _forward(state.model, candidate, rgb, spectrum, local, return_aux=needs_aux)
        logits, visual, spectral = output if needs_aux else (output, None, None)
        loss = _loss(state.model, candidate, logits, labels, visual, spectral)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss for {candidate.candidate_id}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), GRADIENT_CLIP)
        state.optimizer.step()
        state.completed_steps += 1
        loss_sum += float(loss.detach().item()) * len(selected)
        seen += len(selected)
    state.scheduler.step()
    return loss_sum / seen


def _predict_probabilities(state: TrainState, candidate: ScreeningCandidate, sample_ids: Sequence[str], arrays: HistologyArrays, id_to_index: Mapping[str, int], device: torch.device) -> list[tuple[str, float]]:
    sample_indices = _indices(sample_ids, id_to_index)
    result: list[tuple[str, float]] = []
    state.model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_indices), EVALUATION_BATCH_SIZE):
            selected = sample_indices[start:start + EVALUATION_BATCH_SIZE]
            rgb = torch.from_numpy(arrays.rgb[selected]).to(device, non_blocking=True)
            spectrum = torch.from_numpy(arrays.global_spectrum[selected]).to(device, non_blocking=True)
            local = torch.from_numpy(arrays.local_tokens[selected]).to(device, non_blocking=True)
            logits = _forward(state.model, candidate, rgb, spectrum, local, return_aux=False)
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            result.extend((arrays.sample_ids[int(index)], float(probability)) for index, probability in zip(selected, probabilities))
    return result


def _checkpoint(state: TrainState, candidate: ScreeningCandidate, *, protocol_hash: str, fold_id: str, seed: int, epoch: int, identity_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "task_id": "15.13", "candidate_id": candidate.candidate_id,
        "candidate_config_hash": candidate.config_hash, "protocol_hash": protocol_hash,
        "fold_id": fold_id, "seed": seed, "completed_epoch": epoch,
        "completed_training_steps": state.completed_steps, "identity_hash": identity_hash,
        "model": state.model.state_dict(), "optimizer": state.optimizer.state_dict(), "scheduler": state.scheduler.state_dict(),
    }


def _load_checkpoint(path: Path, state: TrainState, *, identity_hash: str, device: torch.device) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("identity_hash") != identity_hash:
        raise RuntimeError("recovery identity mismatch")
    state.model.load_state_dict(payload["model"], strict=True)
    state.optimizer.load_state_dict(payload["optimizer"])
    state.scheduler.load_state_dict(payload["scheduler"])
    state.completed_steps = int(payload["completed_training_steps"])
    return int(payload["completed_epoch"])


def _prediction_artifact(rows: Sequence[Mapping[str, Any]], *, artifact_id: str, run_id: str, normalize_context: str | None = None) -> PredictionArtifact:
    records = tuple(PredictionRecord(
        key=PredictionKey(
            sample_id=str(row["sample_id"]),
            fold_id=normalize_context or str(row["fold_id"]),
            seed=int(row["seed"]),
            inference_branch=str(row["inference_branch"]),
        ),
        target_name="tumor", y_true=int(row["y_true"]),
        y_pred=int(float(row["probability"]) >= PRIMARY_THRESHOLD), probability=float(row["probability"]),
    ) for row in rows)
    expected = tuple(sorted({record.key.sample_id for record in records}, key=str.encode))
    return PredictionArtifact.create(artifact_id=artifact_id, run_id=run_id, records=records, expected_sample_ids=expected, complete=True)


def _metric_payload(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    normalized = [{**row, "fold_id": label} for row in rows]
    artifact = _prediction_artifact(normalized, artifact_id=f"metric-{_safe_id(label)}", run_id=f"metric-{_safe_id(label)}")
    metrics = binary_metrics(artifact, _metric_schema()).model_dump(mode="json")
    return {
        **metrics["metrics"], "confusion_matrix": metrics["confusion_matrices"]["confusion_matrix"],
        "source_prediction_hash": metrics["source_prediction_hash"], "sample_count": len(rows),
    }


def _fit_secondary(rows: Sequence[Mapping[str, Any]], outer: Any, dataset_hash: str, protocol_hash: str) -> dict[str, Any]:
    normalized = [{**row, "fold_id": f"{outer.fold_id}.inner-oof"} for row in rows]
    artifact = _prediction_artifact(normalized, artifact_id=f"inner-{outer.fold_id}", run_id=f"inner-{outer.fold_id}")
    manager = PreprocessingManager()
    from benchmark.schema import FitBoundary, FitBoundaryScope
    boundary = FitBoundary(fold_id=outer.fold_id, scope=FitBoundaryScope.OUTER_TRAIN, allowed_fit_sample_ids=outer.outer_train_ids, outer_test_sample_ids=outer.outer_test_ids)
    probabilities = tuple(float(row["probability"]) for row in normalized)
    labels = tuple(int(row["y_true"]) for row in normalized)
    ids = tuple(str(row["sample_id"]) for row in normalized)
    threshold = manager.fit_threshold(probabilities, labels, ids, boundary, dataset_hash=dataset_hash, protocol_hash=protocol_hash)
    temperature = manager.fit_calibration(probabilities, labels, ids, boundary, dataset_hash=dataset_hash, protocol_hash=protocol_hash, input_kind="probability")
    return {
        "primary": {"label": "primary.fixed-threshold-0.5", "threshold": 0.5},
        "fitted_threshold": {"label": "secondary.inner-oof-fitted-threshold", "value": float(threshold.state["threshold"]), "artifact": threshold.model_dump(mode="json")},
        "fitted_temperature": {"label": "secondary.inner-oof-fitted-temperature", "value": float(temperature.state["temperature"]), "artifact": temperature.model_dump(mode="json")},
        "source_prediction_hash": artifact.content_hash,
    }


def _run_inner_selection_candidate(execution_root: Path, candidate: ScreeningCandidate, folds: Sequence[ScreeningFold], arrays: HistologyArrays, id_to_index: Mapping[str, int], protocol_hash: str, template: Any, environment: Any, registry: Any, device: torch.device, overall_started_ns: int) -> dict[str, Any]:
    root = execution_root / "inner-selection" / candidate.safe_id
    root.mkdir(parents=True, exist_ok=True)
    completed_result_path = root / "result.json"
    if completed_result_path.exists():
        completed_result = _read_json(completed_result_path)
        if (
            completed_result.get("candidate_config_hash") != candidate.config_hash
            or int(completed_result.get("attained_epoch", -1)) != INNER_SELECTION_MAX_EPOCHS
        ):
            raise RuntimeError("completed inner-selection result identity mismatch")
        return completed_result
    states = [_state(candidate, INNER_SELECTION_SEED, index + 1, INNER_SELECTION_MAX_EPOCHS, registry, device) for index, _ in enumerate(folds)]
    identity = {
        "task_id": "15.13", "stage": "outer-fold-inner-selection", "dataset_hash": arrays.cache_hashes,
        "protocol_hash": protocol_hash, "candidate_id": candidate.candidate_id,
        "candidate_config_hash": candidate.config_hash, "seed": INNER_SELECTION_SEED,
        "max_epochs": INNER_SELECTION_MAX_EPOCHS, "fold_ids": [fold.fold_id for fold in folds],
        "environment_lock_hash": environment.lock_hash,
    }
    identity_hash = hash_object(identity)
    _publish_json_once(root / "frozen-identity.json", {**identity, "identity_hash": identity_hash})
    marker_path = root / "common-recovery.json"
    completed = 0
    history: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    if marker_path.exists():
        marker = _read_json(marker_path)
        if marker.get("identity_hash") != identity_hash:
            raise RuntimeError("inner-selection recovery identity mismatch")
        completed = int(marker["completed_epoch"])
        history = list(marker["history"])
        best = dict(marker["best_by_outer"])
        for state, fold in zip(states, folds):
            checkpoint = root / "folds" / fold.fold_id / "recovery" / f"slot-{completed % 2}.pt"
            if _load_checkpoint(checkpoint, state, identity_hash=identity_hash, device=device) != completed:
                raise RuntimeError("inner-selection fold recovery boundary mismatch")
    torch.cuda.reset_peak_memory_stats(0)
    for epoch in range(completed + 1, INNER_SELECTION_MAX_EPOCHS + 1):
        losses = {}
        rows: list[dict[str, Any]] = []
        for ordinal, (state, fold) in enumerate(zip(states, folds), start=1):
            losses[fold.fold_id] = _train_epoch(state, candidate, fold.train_ids, arrays, id_to_index, INNER_SELECTION_SEED, ordinal, epoch, device)
            payload = _checkpoint(state, candidate, protocol_hash=protocol_hash, fold_id=fold.fold_id, seed=INNER_SELECTION_SEED, epoch=epoch, identity_hash=identity_hash)
            _atomic_torch(root / "folds" / fold.fold_id / "recovery" / f"slot-{epoch % 2}.pt", payload)
            for sample_id, probability in _predict_probabilities(state, candidate, fold.validation_ids, arrays, id_to_index, device):
                if sample_id in set(fold.forbidden_outer_test_ids):
                    raise RuntimeError("outer-test sample reached inner selection")
                index = id_to_index[sample_id]
                rows.append({
                    "sample_id": sample_id, "group_id": arrays.group_by_sample[sample_id],
                    "outer_fold_id": fold.outer_fold_id, "fold_id": fold.fold_id,
                    "seed": INNER_SELECTION_SEED, "inference_branch": candidate.inference_branch,
                    "y_true": int(arrays.labels[index]), "probability": probability,
                })
        epoch_scores: dict[str, float] = {}
        for outer_id in sorted({fold.outer_fold_id for fold in folds}, key=str.encode):
            outer_rows = [row for row in rows if row["outer_fold_id"] == outer_id]
            metric = _metric_payload(outer_rows, f"{outer_id}.inner-oof")
            score = metric["balanced_accuracy"]["value"]
            if score is None:
                raise RuntimeError(f"undefined inner balanced accuracy for {outer_id}")
            epoch_scores[outer_id] = float(score)
            previous = best.get(outer_id)
            if previous is None or float(score) > float(previous["score"]):
                selected_payload = {
                    "candidate_id": candidate.candidate_id, "outer_fold_id": outer_id,
                    "epoch": epoch, "score": float(score), "records": outer_rows,
                    "source_prediction_hash": hash_object(outer_rows),
                }
                _atomic_json(root / "selected" / f"{outer_id}.json", selected_payload)
                best[outer_id] = {
                    "epoch": epoch, "score": float(score), "source_prediction_hash": selected_payload["source_prediction_hash"],
                    "source_sample_ids": tuple(sorted((row["sample_id"] for row in outer_rows), key=str.encode)),
                }
        history.append({"epoch": epoch, "scores_by_outer": epoch_scores, "losses": losses})
        _atomic_json(marker_path, {"identity_hash": identity_hash, "completed_epoch": epoch, "history": history, "best_by_outer": best})
        measurements = RuntimeMeasurements(
            completed_epochs=epoch, completed_training_steps=max(state.completed_steps for state in states),
            wall_clock_duration_ms=int((time.monotonic_ns() - overall_started_ns) // 1_000_000),
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)), device_identity=environment.device_identity,
            numeric_precision="float32", parameter_count=sum(parameter.numel() for parameter in states[0].model.parameters()),
            input_volume_bytes=arrays.input_volume_bytes,
        )
        assessment = assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch)
        if assessment.hard_exceeded:
            raise RuntimeError(f"inner selection hard budget exceeded: {assessment.model_dump(mode='json')}")
        print(json.dumps({"stage": "inner_selection", "candidate_id": candidate.candidate_id, "epoch": epoch, "max_epoch": INNER_SELECTION_MAX_EPOCHS, "scores_by_outer": epoch_scores}), flush=True)
    for state, fold in zip(states, folds):
        payload = _checkpoint(state, candidate, protocol_hash=protocol_hash, fold_id=fold.fold_id, seed=INNER_SELECTION_SEED, epoch=INNER_SELECTION_MAX_EPOCHS, identity_hash=identity_hash)
        _atomic_torch(root / "folds" / fold.fold_id / "checkpoints" / "last.pt", payload)
    result = {
        "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
        "attained_epoch": INNER_SELECTION_MAX_EPOCHS, "best_by_outer": best,
        "completed_training_steps_per_inner_fold": {fold.fold_id: state.completed_steps for state, fold in zip(states, folds)},
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "parameter_count": sum(parameter.numel() for parameter in states[0].model.parameters()),
    }
    _publish_json_once(root / "result.json", result)
    del states
    torch.cuda.empty_cache()
    return result


def _evaluate_job(execution_root: Path, protocol_kind: str, protocol_hash: str, candidate: ScreeningCandidate, fold: EvalFold, seed: int, epochs: int, arrays: HistologyArrays, id_to_index: Mapping[str, int], secondary: Mapping[str, Any] | None, plan_hash: str, template: Any, environment: Any, registry: Any, device: torch.device, overall_started_ns: int) -> dict[str, Any]:
    root = execution_root / protocol_kind / "jobs" / candidate.safe_id / f"seed-{seed}" / fold.fold_id
    result_path = root / "result.json"
    if result_path.exists():
        result = _read_json(result_path)
        prediction_path = execution_root / str(result["prediction_relative_to_execution"])
        if hash_file(prediction_path) != result["prediction_file_hash"]:
            raise RuntimeError("completed job prediction hash mismatch")
        return result
    fold_ordinal = int(hashlib.sha256(fold.fold_id.encode("utf-8")).hexdigest()[:8], 16)
    state = _state(candidate, seed, fold_ordinal, epochs, registry, device)
    identity = {
        "dataset_hash": MANIFEST_PATH.stem, "protocol_hash": protocol_hash,
        "model_key": candidate.model_key, "candidate_id": candidate.candidate_id,
        "candidate_config_hash": candidate.config_hash, "fold_id": fold.fold_id,
        "seed": seed, "epochs": epochs, "plan_hash": plan_hash,
        "environment_lock_hash": environment.lock_hash, "numeric_precision": "float32",
    }
    identity_hash = hash_object(identity)
    run_id = f"task-15-13-{protocol_kind}-{candidate.safe_id}-seed-{seed}-{fold.fold_id}-{identity_hash[:12]}"
    _publish_json_once(root / "frozen-run-identity.json", {**identity, "identity_hash": identity_hash, "run_id": run_id})
    marker_path = root / "recovery" / "marker.json"
    completed = 0
    if marker_path.exists():
        marker = _read_json(marker_path)
        if marker.get("identity_hash") != identity_hash:
            raise RuntimeError("job recovery marker identity mismatch")
        completed = int(marker["completed_epoch"])
        checkpoint_path = root / "recovery" / f"slot-{completed % 2}.pt"
        if _load_checkpoint(checkpoint_path, state, identity_hash=identity_hash, device=device) != completed:
            raise RuntimeError("job recovery checkpoint boundary mismatch")
    started_ns = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats(0)
    last_loss = None
    for epoch in range(completed + 1, epochs + 1):
        last_loss = _train_epoch(state, candidate, fold.train_ids, arrays, id_to_index, seed, fold_ordinal, epoch, device)
        payload = _checkpoint(state, candidate, protocol_hash=protocol_hash, fold_id=fold.fold_id, seed=seed, epoch=epoch, identity_hash=identity_hash)
        _atomic_torch(root / "recovery" / f"slot-{epoch % 2}.pt", payload)
        _atomic_json(marker_path, {"identity_hash": identity_hash, "completed_epoch": epoch, "completed_training_steps": state.completed_steps, "last_loss": last_loss})
        measurements = RuntimeMeasurements(
            completed_epochs=epoch, completed_training_steps=state.completed_steps,
            wall_clock_duration_ms=int((time.monotonic_ns() - overall_started_ns) // 1_000_000),
            peak_device_memory_bytes=int(torch.cuda.max_memory_allocated(0)), device_identity=environment.device_identity,
            numeric_precision="float32", parameter_count=sum(parameter.numel() for parameter in state.model.parameters()),
            input_volume_bytes=arrays.input_volume_bytes,
        )
        if assess_resource_budget(template.resource_budget, measurements, checkpoint_boundary=epoch).hard_exceeded:
            raise RuntimeError("strict job hard budget exceeded")
    final_payload = _checkpoint(state, candidate, protocol_hash=protocol_hash, fold_id=fold.fold_id, seed=seed, epoch=epochs, identity_hash=identity_hash)
    _atomic_torch(root / "checkpoints" / "selected.pt", final_payload)
    _atomic_torch(root / "checkpoints" / "last.pt", final_payload)

    # Probabilities are generated before labels are unlocked.  The immutable
    # per-fold decision/policy and secondary fit identities already exist.
    probabilities = _predict_probabilities(state, candidate, fold.test_ids, arrays, id_to_index, device)
    access = {
        "protocol_kind": protocol_kind, "fold_id": fold.fold_id, "seed": seed,
        "frozen_plan_hash": plan_hash, "candidate_id": candidate.candidate_id,
        "fixed_epochs": epochs, "secondary_fitted_before_access": secondary is not None,
        "outer_label_accessed_utc": _utc_now(), "successful_evaluations_consumed": 1,
    }
    _publish_json_once(root / "outer-label-access.json", access)
    rows = []
    for sample_id, probability in probabilities:
        index = id_to_index[sample_id]
        row = {
            "sample_id": sample_id, "group_id": arrays.group_by_sample[sample_id],
            "fold_id": fold.fold_id, "seed": seed, "inference_branch": candidate.inference_branch,
            "y_true": int(arrays.labels[index]), "probability": probability,
            "y_pred_fixed_0_5": int(probability >= 0.5),
        }
        if secondary is not None:
            threshold = float(secondary["fitted_threshold"]["value"])
            temperature = float(secondary["fitted_temperature"]["value"])
            clipped = min(max(probability, 1e-7), 1 - 1e-7)
            calibrated = 1.0 / (1.0 + math.exp(-math.log(clipped / (1.0 - clipped)) / temperature))
            row.update({"inner_oof_threshold": threshold, "y_pred_inner_oof_threshold": int(probability >= threshold), "inner_oof_temperature": temperature, "probability_inner_oof_temperature": calibrated})
        rows.append(row)
    if set(row["sample_id"] for row in rows) != set(fold.test_ids) or len(rows) != len(fold.test_ids):
        raise RuntimeError("outer prediction coverage mismatch")
    artifact = _prediction_artifact(rows, artifact_id=f"{run_id}-predictions", run_id=run_id)
    prediction_payload = {"schema_version": "1.0", "artifact": artifact.model_dump(mode="json"), "records_extended": rows}
    prediction_path = root / "predictions.json"
    _publish_json_once(prediction_path, prediction_payload)
    metrics = binary_metrics(artifact, _metric_schema()).model_dump(mode="json")
    runtime = {
        "completed_epochs": epochs, "completed_training_steps": state.completed_steps,
        "wall_clock_duration_ms": int((time.monotonic_ns() - started_ns) // 1_000_000),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        "device_identity": environment.device_identity, "numeric_precision": "float32",
        "parameter_count": sum(parameter.numel() for parameter in state.model.parameters()),
        "input_volume_bytes": arrays.input_volume_bytes,
    }
    result = {
        "status": "complete", "run_id": run_id, "protocol_kind": protocol_kind,
        "candidate_id": candidate.candidate_id, "candidate_config_hash": candidate.config_hash,
        "fold_id": fold.fold_id, "seed": seed, "epochs": epochs,
        "prediction_relative_to_execution": prediction_path.relative_to(execution_root).as_posix(),
        "prediction_content_hash": artifact.content_hash, "prediction_file_hash": hash_file(prediction_path),
        "metrics": metrics, "runtime_measurements": runtime,
        "selected_checkpoint_hash": hash_file(root / "checkpoints" / "selected.pt"),
        "last_checkpoint_hash": hash_file(root / "checkpoints" / "last.pt"),
        "recovery_checkpoint_hash": hash_file(root / "recovery" / f"slot-{epochs % 2}.pt"),
        "successful_outer_evaluations_consumed": 1,
    }
    _publish_json_once(result_path, result)
    del state
    torch.cuda.empty_cache()
    print(json.dumps({"stage": protocol_kind, "candidate_id": candidate.candidate_id, "seed": seed, "fold_id": fold.fold_id, "epochs": epochs}), flush=True)
    return result


def _load_rows(execution_root: Path, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _read_json(execution_root / str(result["prediction_relative_to_execution"]))
    return list(payload["records_extended"])


def _aggregate_protocol(execution_root: Path, protocol_kind: str, candidates: Sequence[ScreeningCandidate], results: Sequence[Mapping[str, Any]], sample_ids: Sequence[str], selected_policy_by_fold: Mapping[str, str], group_by_sample: Mapping[str, str]) -> dict[str, Any]:
    candidate_summaries: dict[str, Any] = {}
    aggregate_root = execution_root / protocol_kind / "aggregates"
    for candidate in candidates:
        candidate_results = [result for result in results if result["candidate_id"] == candidate.candidate_id]
        per_seed = {}
        seed_paths = {}
        consensus_probabilities: dict[str, list[float]] = {sample_id: [] for sample_id in sample_ids}
        for seed in STRICT_SEEDS:
            selected = [result for result in candidate_results if int(result["seed"]) == seed]
            rows = [row for result in selected for row in _load_rows(execution_root, result)]
            if len(rows) != len(sample_ids) or {row["sample_id"] for row in rows} != set(sample_ids):
                raise RuntimeError(f"{protocol_kind}/{candidate.candidate_id}/seed-{seed} OOF coverage mismatch")
            pooled = _metric_payload(rows, f"{protocol_kind}.{candidate.safe_id}.seed-{seed}.pooled")
            per_fold = {fold_id: _metric_payload([row for row in rows if row["fold_id"] == fold_id], f"{protocol_kind}.{candidate.safe_id}.seed-{seed}.{fold_id}") for fold_id in sorted({row["fold_id"] for row in rows}, key=str.encode)}
            fold_macro = {}
            for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity", "ece"):
                values = [payload[metric]["value"] for payload in per_fold.values() if payload[metric]["value"] is not None]
                fold_macro[metric] = {"value": None if not values else float(np.mean(values)), "defined_count": len(values), "total_count": len(per_fold)}
            per_seed[str(seed)] = {"pooled_oof": pooled, "fold_macro": fold_macro, "per_fold": per_fold}
            for row in rows:
                consensus_probabilities[row["sample_id"]].append(float(row["probability"]))
            payload = {
                "schema_version": "1.0", "protocol_kind": protocol_kind, "candidate_id": candidate.candidate_id,
                "seed": seed, "member_run_ids": [result["run_id"] for result in selected],
                "prediction_hashes": [result["prediction_content_hash"] for result in selected],
                "metrics": per_seed[str(seed)], "records": rows,
            }
            payload["aggregate_hash"] = hash_object(payload)
            path = aggregate_root / candidate.safe_id / f"seed-{seed}-oof.json"
            _publish_json_once(path, payload)
            seed_paths[str(seed)] = {"path": path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(path)}
        scalar_aggregate = {}
        for scope in ("pooled_oof", "fold_macro"):
            scalar_aggregate[scope] = {}
            for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity", "ece"):
                values = []
                for seed in STRICT_SEEDS:
                    item = per_seed[str(seed)][scope][metric]
                    value = item["value"] if isinstance(item, dict) else item
                    if value is not None:
                        values.append(float(value))
                scalar_aggregate[scope][metric] = {"seed_mean": None if not values else float(np.mean(values)), "seed_variance": None if len(values) < 2 else float(np.var(values, ddof=1)), "defined_seed_count": len(values)}
        consensus_rows = []
        first_seed_rows = {row["sample_id"]: row for result in candidate_results if int(result["seed"]) == STRICT_SEEDS[0] for row in _load_rows(execution_root, result)}
        for sample_id in sample_ids:
            base = first_seed_rows[sample_id]
            consensus_rows.append({**base, "fold_id": f"{protocol_kind}.seed-consensus", "seed": BOOTSTRAP_SEED, "inference_branch": candidate.inference_branch + ".seed-mean", "probability": float(np.mean(consensus_probabilities[sample_id]))})
        consensus_artifact = _prediction_artifact(consensus_rows, artifact_id=f"{protocol_kind}-{candidate.safe_id}-seed-consensus", run_id=f"{protocol_kind}-{candidate.safe_id}-seed-consensus")
        consensus_path = aggregate_root / candidate.safe_id / "seed-consensus-oof.json"
        _publish_json_once(consensus_path, {"schema_version": "1.0", "artifact": consensus_artifact.model_dump(mode="json"), "records_extended": consensus_rows})
        candidate_summaries[candidate.candidate_id] = {
            "configuration": {**asdict(candidate), "config_hash": candidate.config_hash},
            "per_seed_metrics": per_seed, "aggregate_metrics": scalar_aggregate,
            "seed_oof_artifacts": seed_paths,
            "consensus_artifact": consensus_artifact,
            "consensus_path": consensus_path.relative_to(WORKSPACE).as_posix(),
            "member_run_ids": [result["run_id"] for result in candidate_results],
            "prediction_hashes": [result["prediction_content_hash"] for result in candidate_results],
            "successful_outer_evaluations_consumed_per_model": len(candidate_results),
        }

    selected_consensus_probabilities: dict[str, list[float]] = {sample_id: [] for sample_id in sample_ids}
    selected_member_ids = []
    for seed in STRICT_SEEDS:
        for fold_id, candidate_id in selected_policy_by_fold.items():
            result = next(item for item in results if item["candidate_id"] == candidate_id and int(item["seed"]) == seed and item["fold_id"] == fold_id)
            selected_member_ids.append(result["run_id"])
            for row in _load_rows(execution_root, result):
                selected_consensus_probabilities[row["sample_id"]].append(float(row["probability"]))
    selected_rows = []
    labels = {sample_id: None for sample_id in sample_ids}
    fold_for_sample = {}
    for result in results:
        if result["fold_id"] in selected_policy_by_fold and result["candidate_id"] == selected_policy_by_fold[result["fold_id"]] and int(result["seed"]) == STRICT_SEEDS[0]:
            for row in _load_rows(execution_root, result):
                labels[row["sample_id"]] = int(row["y_true"])
                fold_for_sample[row["sample_id"]] = row["fold_id"]
    for sample_id in sample_ids:
        selected_rows.append({
            "sample_id": sample_id, "group_id": group_by_sample[sample_id], "fold_id": f"{protocol_kind}.selected-policy-consensus",
            "seed": BOOTSTRAP_SEED, "inference_branch": "per-outer-fold-selected.seed-mean",
            "y_true": labels[sample_id], "probability": float(np.mean(selected_consensus_probabilities[sample_id])),
            "source_outer_fold_id": fold_for_sample[sample_id],
        })
    selected_artifact = _prediction_artifact(selected_rows, artifact_id=f"{protocol_kind}-selected-policy-consensus", run_id=f"{protocol_kind}-selected-policy-consensus")
    selected_metrics = binary_metrics(selected_artifact, _metric_schema()).model_dump(mode="json")
    statistics_root = aggregate_root / "selected-policy-statistics"
    uncertainty = {}
    for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity", "ece"):
        statistic_path = statistics_root / f"bootstrap-{metric}.json"
        if statistic_path.exists():
            uncertainty[metric] = _read_json(statistic_path)
        else:
            interval = patient_bootstrap(selected_artifact, metric, group_by_sample_id=group_by_sample, metric_schema=_metric_schema(), seed=BOOTSTRAP_SEED)
            uncertainty[metric] = interval.model_dump(mode="json")
            _publish_json_once(statistic_path, uncertainty[metric])
    paired = {}
    for candidate_id, summary in candidate_summaries.items():
        statistic_path = statistics_root / f"paired-balanced-accuracy-{_safe_id(candidate_id)}.json"
        if statistic_path.exists():
            paired[candidate_id] = _read_json(statistic_path)
        else:
            comparison = paired_patient_compare(
                selected_artifact, summary["consensus_artifact"], "balanced_accuracy",
                candidate_group_by_sample_id=group_by_sample, control_group_by_sample_id=group_by_sample,
                metric_schema=_metric_schema(), seed=BOOTSTRAP_SEED,
            )
            paired[candidate_id] = comparison.model_dump(mode="json")
            _publish_json_once(statistic_path, paired[candidate_id])
        summary.pop("consensus_artifact")
    selected_path = aggregate_root / "selected-policy-seed-consensus-oof.json"
    _publish_json_once(selected_path, {"schema_version": "1.0", "artifact": selected_artifact.model_dump(mode="json"), "records_extended": selected_rows, "selected_policy_by_fold": dict(selected_policy_by_fold)})
    return {
        "candidate_control_table": candidate_summaries,
        "selected_policy": {
            "selected_candidate_by_fold": dict(selected_policy_by_fold), "member_run_ids": selected_member_ids,
            "consensus_metrics": selected_metrics, "patient_bootstrap_10000": uncertainty,
            "paired_balanced_accuracy_10000": paired,
            "prediction_artifact": selected_path.relative_to(WORKSPACE).as_posix(), "prediction_hash": selected_artifact.content_hash,
        },
    }


def execute_strict(*, resume_execution_id: str | None = None) -> dict[str, Any]:
    for path in (WORKSPACE, DATA_ROOT, MANIFEST_PATH, TEMPLATE_PATH, SCREENING_SUMMARY, OUTPUT_ROOT):
        if PureWindowsPath(str(path)).drive.casefold() != "g:":
            raise RuntimeError(f"task 15.13 is restricted to approved G: roots: {path}")
    template = load_fast_lane_template(TEMPLATE_PATH)
    if template.stage is not FastLaneStage.STRICT or tuple(template.seed_set) != STRICT_SEEDS:
        raise RuntimeError("task 15.13 requires the immutable strict template and seeds 7/42/123")
    environment = validate_live_environment(template)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    _set_determinism(INNER_SELECTION_SEED)
    overall_started_ns = time.monotonic_ns()

    summary = _read_json(SCREENING_SUMMARY)
    screening_report = WORKSPACE / str(summary["report"])
    screening_decision_path = screening_report.parent / "frozen-selection-decision.json"
    screening_plan_path = screening_report.parent / "frozen-screening-plan.json"
    screening_decision = _read_json(screening_decision_path)
    candidates = validate_screening_checkpoint(summary, screening_decision)
    manifest = _load_manifest()
    group_by_sample = {record.sample_id: str(record.group_id) for record in manifest.records}

    execution_id = resume_execution_id or datetime.now(timezone.utc).strftime("task-15-13-%Y%m%dT%H%M%S%fZ")
    execution_root = OUTPUT_ROOT / "executions" / execution_id
    if resume_execution_id:
        if not execution_root.is_dir():
            raise RuntimeError("requested task-15.13 execution does not exist")
    else:
        execution_root.mkdir(parents=True, exist_ok=False)
    store = FailureAtomicArtifactStore(execution_root / "protocol-store")
    nested_protocol = HistologyNested5Planner(ProtocolPlanner(store)).plan(manifest, seed_set=STRICT_SEEDS)
    screening_folds = build_screening_folds(nested_protocol.folds, group_by_sample)
    arrays = _load_arrays()
    if arrays.sample_ids != tuple(manifest.canonical_order) or dict(arrays.group_by_sample) != group_by_sample:
        raise RuntimeError("cache order or patient mapping differs from frozen manifest")
    id_to_index = {sample_id: index for index, sample_id in enumerate(arrays.sample_ids)}

    strict_config = {
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "seed_set": list(STRICT_SEEDS), "inner_selection_seed": INNER_SELECTION_SEED,
        "inner_selection_max_epochs": INNER_SELECTION_MAX_EPOCHS,
        "primary_threshold": 0.5, "secondary_fit": "separate per outer fold from selected-candidate complete inner OOF",
        "preprocessing_track": "cache_bound_historical_bytes_with_incomplete_creation_lineage",
        "raw_lineage_track_status": "blocked_not_executed",
        "raw_cube_total_bytes": RAW_CUBE_TOTAL_BYTES,
        "raw_regeneration_blocker": "579.57 GiB raw cubes exceed frozen 8,000,000,000-byte input-volume budget",
        "nested_selection": "separate candidate/epoch decision per outer fold from only its five inner-patient OOF folds",
        "nested_control_budget": "all controls train the selected outer-fold epoch",
        "lopo_epoch_policy": "median nested selected epoch frozen before LOPO label access",
        "device": "cuda:0", "process_count": 1, "precision": "float32", "deterministic": True,
    }
    base_plan = {
        "schema_version": "1.0", "task_id": "15.13", "execution_id": execution_id,
        "created_utc": _utc_now(), "dataset_hash": manifest.dataset_hash,
        "nested_protocol_key": nested_protocol.protocol_key, "nested_protocol_hash": nested_protocol.protocol_hash,
        "strict_config": strict_config, "strict_config_hash": hash_object(strict_config),
        "candidate_table": [{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates],
        "candidate_space_hash": hash_object([{**asdict(candidate), "config_hash": candidate.config_hash} for candidate in candidates]),
        "budget": template.resource_budget.model_dump(mode="json"), "budget_hash": hash_object(template.resource_budget.model_dump(mode="json")),
        "template": {"path": TEMPLATE_PATH.relative_to(WORKSPACE).as_posix(), "hash": hash_file(TEMPLATE_PATH), "template_hash": template.template_hash},
        "screening_source": {"summary": SCREENING_SUMMARY.relative_to(WORKSPACE).as_posix(), "summary_hash": hash_file(SCREENING_SUMMARY), "plan_hash": hash_file(screening_plan_path), "decision_hash": hash_file(screening_decision_path), "global_winner_adopted_for_nested": False},
        "scope": {"write_root": "runs_v2/strict/histology", "hyperleaf_official_evaluation_allowed": False, "read_only_roots": ["src", "runs", "dataset", "caches"]},
    }
    initial_plan_path = execution_root / "frozen-initial-plan.json"
    if resume_execution_id:
        existing = _read_json(initial_plan_path)
        base_plan["created_utc"] = existing["created_utc"]
        if _json_bytes(existing) != _json_bytes(base_plan):
            raise RuntimeError("resume plan differs from frozen task-15.13 plan")
    else:
        _publish_json_once(initial_plan_path, base_plan)
        _publish_json_once(execution_root / "environment-lock.json", environment.model_dump(mode="json"))
        _atomic_bytes(execution_root / "nested-protocol.canonical.json", canonical_serialize(nested_protocol))

    readonly_before = {
        "src": _tree_fingerprint(WORKSPACE / "src"), "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "dataset": _tree_fingerprint(DATA_ROOT), "manifest": hash_file(MANIFEST_PATH),
        "screening_summary": hash_file(SCREENING_SUMMARY), "screening_plan": hash_file(screening_plan_path),
        "screening_decision": hash_file(screening_decision_path),
    }
    registry = create_histology_model_registry()
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in candidates:
        result = _run_inner_selection_candidate(execution_root, candidate, screening_folds, arrays, id_to_index, nested_protocol.protocol_hash, template, environment, registry, device, overall_started_ns)
        evidence[candidate.candidate_id] = dict(result["best_by_outer"])
    outer_decisions_path = execution_root / "frozen-outer-decisions.json"
    secondary_path = execution_root / "fitted-secondary-by-outer.json"
    if outer_decisions_path.exists() or secondary_path.exists():
        if not outer_decisions_path.exists() or not secondary_path.exists():
            raise RuntimeError("outer decision and secondary artifacts must recover together")
        outer_decisions = _read_json(outer_decisions_path)
        secondary_by_outer = _read_json(secondary_path)
        for outer in nested_protocol.folds:
            decision = outer_decisions[outer.fold_id]
            if set(decision["source_sample_ids"]) != set(outer.outer_train_ids) or set(decision["source_sample_ids"]) & set(outer.outer_test_ids):
                raise RuntimeError(f"recovered selection source boundary violation in {outer.fold_id}")
    else:
        outer_decisions = choose_outer_decisions(evidence)
        secondary_by_outer = {}
        for outer in nested_protocol.folds:
            decision = outer_decisions[outer.fold_id]
            if set(decision["source_sample_ids"]) != set(outer.outer_train_ids) or set(decision["source_sample_ids"]) & set(outer.outer_test_ids):
                raise RuntimeError(f"selection source boundary violation in {outer.fold_id}")
            selected_candidate = _candidate_map()[decision["selected_candidate"]]
            selected_payload = _read_json(execution_root / "inner-selection" / selected_candidate.safe_id / "selected" / f"{outer.fold_id}.json")
            secondary_by_outer[outer.fold_id] = _fit_secondary(selected_payload["records"], outer, manifest.dataset_hash, nested_protocol.protocol_hash)
            decision["secondary"] = secondary_by_outer[outer.fold_id]
            decision["frozen_utc"] = _utc_now()
            decision["decision_hash"] = hash_object(decision)
        _publish_json_once(outer_decisions_path, outer_decisions)
        _publish_json_once(secondary_path, secondary_by_outer)

    lopo_policy_path = execution_root / "frozen-lopo-policy.json"
    if lopo_policy_path.exists():
        lopo_policy = _read_json(lopo_policy_path)
    else:
        lopo_policy = freeze_lopo_policy(outer_decisions)
        _publish_json_once(lopo_policy_path, lopo_policy)
    lopo_epoch_policy = HistologyFixedEpochPolicy(int(lopo_policy["fixed_epoch"]))
    lopo_protocol = HistologyLopoFixedPlanner(ProtocolPlanner(store), epoch_policy=lopo_epoch_policy).plan(manifest, seed_set=STRICT_SEEDS)
    _atomic_bytes(execution_root / "lopo-protocol.canonical.json", canonical_serialize(lopo_protocol))
    final_plan = {
        **base_plan, "outer_decisions_hash": hash_file(execution_root / "frozen-outer-decisions.json"),
        "secondary_hash": hash_file(execution_root / "fitted-secondary-by-outer.json"),
        "lopo_policy": lopo_policy, "lopo_protocol_key": lopo_protocol.protocol_key,
        "lopo_protocol_hash": lopo_protocol.protocol_hash,
    }
    final_plan["final_plan_hash"] = hash_object(final_plan)
    _publish_json_once(execution_root / "frozen-final-plan.json", final_plan)
    plan_hash = final_plan["final_plan_hash"]

    nested_results = []
    failures = []
    for candidate in candidates:
        for seed in STRICT_SEEDS:
            for outer in nested_protocol.folds:
                fold = EvalFold(outer.fold_id, outer.outer_train_ids, outer.outer_test_ids)
                try:
                    nested_results.append(_evaluate_job(execution_root, "nested", nested_protocol.protocol_hash, candidate, fold, seed, int(outer_decisions[outer.fold_id]["selected_epoch"]), arrays, id_to_index, secondary_by_outer[outer.fold_id], plan_hash, template, environment, registry, device, overall_started_ns))
                except Exception as exc:
                    failure = {"protocol": "nested", "candidate_id": candidate.candidate_id, "seed": seed, "fold_id": fold.fold_id, "error": repr(exc), "created_utc": _utc_now()}
                    failures.append(failure)
                    _append_jsonl(execution_root / "negative-evidence.jsonl", failure)
                    _append_jsonl(OUTPUT_ROOT / "negative-evidence.jsonl", failure)
                    print(json.dumps({"stage": "job_failed", **failure}), file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"nested strict execution has {len(failures)} failed jobs; resume after correction")

    lopo_candidate_id = str(lopo_policy["selected_candidate"])
    lopo_results = []
    for candidate in candidates:
        for seed in STRICT_SEEDS:
            for outer in lopo_protocol.folds:
                fold = EvalFold(outer.fold_id, outer.outer_train_ids, outer.outer_test_ids)
                try:
                    lopo_results.append(_evaluate_job(execution_root, "lopo", lopo_protocol.protocol_hash, candidate, fold, seed, int(lopo_policy["fixed_epoch"]), arrays, id_to_index, None, plan_hash, template, environment, registry, device, overall_started_ns))
                except Exception as exc:
                    failure = {"protocol": "lopo", "candidate_id": candidate.candidate_id, "seed": seed, "fold_id": fold.fold_id, "error": repr(exc), "created_utc": _utc_now()}
                    failures.append(failure)
                    _append_jsonl(execution_root / "negative-evidence.jsonl", failure)
                    _append_jsonl(OUTPUT_ROOT / "negative-evidence.jsonl", failure)
                    print(json.dumps({"stage": "job_failed", **failure}), file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"LOPO strict execution has {len(failures)} failed jobs; resume after correction")

    nested_selected = {fold_id: str(decision["selected_candidate"]) for fold_id, decision in outer_decisions.items()}
    nested_aggregate = _aggregate_protocol(execution_root, "nested", candidates, nested_results, arrays.sample_ids, nested_selected, group_by_sample)
    lopo_selected = {outer.fold_id: lopo_candidate_id for outer in lopo_protocol.folds}
    lopo_aggregate = _aggregate_protocol(execution_root, "lopo", candidates, lopo_results, arrays.sample_ids, lopo_selected, group_by_sample)

    claim = ClaimGate.classify(ClaimEvidence(
        claim_id=f"{execution_id}-cache-bound-strict", implementation_kind=ImplementationKind.IN_REPO,
        detected_leakage=False, model_identity_mismatch=False, remote_sensing_sanity_track=False,
        random_split=False, outer_data_used_for_model_selection=False, single_run=False,
        outer_data_used_for_ensemble_weight_search=False, outer_data_used_for_threshold_search=False,
        strict_protocol_complete=True, provenance_complete=False, seed_set_complete=True, folds_complete=True,
        matched_input_control_required=True, matched_input_control_complete=True,
        prediction_artifact_complete=True, unified_runner_reproduced=True,
        score=nested_aggregate["selected_policy"]["consensus_metrics"]["metrics"]["balanced_accuracy"]["value"],
    ))
    promotion = {
        "status": claim.status.value, "decision": claim.model_dump(mode="json"),
        "nested_outer_reselection_from_outer_labels": False, "lopo_reselection_from_lopo_labels": False,
        "cache_bound_track_complete": True, "raw_lineage_track_complete": False,
        "primary_strict_blocker": "historical local-token and spatial-mean cache creation lineage is incomplete",
        "raw_regeneration_required_bytes": RAW_CUBE_TOTAL_BYTES,
        "raw_regeneration_budget_bytes": next(limit.value for limit in template.resource_budget.limits if str(limit.field) == "input_volume_bytes"),
    }
    _publish_json_once(execution_root / "promotion-evaluation.json", promotion)

    # Append negative evidence when any fixed control exceeds the selected policy.
    selected_score = float(nested_aggregate["selected_policy"]["consensus_metrics"]["metrics"]["balanced_accuracy"]["value"])
    for candidate_id, candidate_summary in nested_aggregate["candidate_control_table"].items():
        control_score = float(candidate_summary["aggregate_metrics"]["pooled_oof"]["balanced_accuracy"]["seed_mean"])
        if control_score > selected_score:
            record = {
                "schema_version": "1.0", "task_id": "15.13", "evidence_id": f"{execution_id}-nested-{_safe_id(candidate_id)}-outperforms-selected",
                "reason": "selected_policy_underperformance", "candidate_id": candidate_id,
                "details": {"selected_policy_balanced_accuracy": selected_score, "control_seed_mean_balanced_accuracy": control_score},
                "created_utc": _utc_now(),
            }
            _append_jsonl(execution_root / "negative-evidence.jsonl", record)
            _append_jsonl(OUTPUT_ROOT / "negative-evidence.jsonl", record)

    readonly_after = {
        "src": _tree_fingerprint(WORKSPACE / "src"), "legacy_runs": _tree_fingerprint(WORKSPACE / "runs"),
        "dataset": _tree_fingerprint(DATA_ROOT), "manifest": hash_file(MANIFEST_PATH),
        "screening_summary": hash_file(SCREENING_SUMMARY), "screening_plan": hash_file(screening_plan_path),
        "screening_decision": hash_file(screening_decision_path),
    }
    if readonly_before != readonly_after:
        raise RuntimeError("read-only source, legacy run, dataset, cache, or screening evidence changed")
    elapsed_ms = int((time.monotonic_ns() - overall_started_ns) // 1_000_000)
    all_results = nested_results + lopo_results
    report = {
        "schema_version": "1.0", "task_id": "15.13", "status": "complete_cache_bound",
        "execution_id": execution_id, "command": COMMAND_EXECUTE, "completed_utc": _utc_now(),
        "recovery_action": "no active duplicate or prior strict execution found; started a new frozen execution and used per-job epoch/patient recovery boundaries",
        "outer_fold_decisions": outer_decisions, "lopo_policy": lopo_policy,
        "nested": nested_aggregate, "lopo": lopo_aggregate,
        "promotion_evaluation": promotion,
        "identity": {
            "dataset_hash": manifest.dataset_hash, "nested_protocol_hash": nested_protocol.protocol_hash,
            "lopo_protocol_hash": lopo_protocol.protocol_hash, "strict_config_hash": base_plan["strict_config_hash"],
            "candidate_space_hash": base_plan["candidate_space_hash"], "final_plan_hash": plan_hash,
            "budget_hash": base_plan["budget_hash"], "environment_lock_hash": environment.lock_hash,
            "runner_hash": hash_file(Path(__file__)), "model_source_hash": hash_file(WORKSPACE / "benchmark" / "models" / "histology.py"),
            "screening_decision_hash": hash_file(screening_decision_path),
        },
        "resource_use": {
            "wall_clock_duration_ms": elapsed_ms,
            "peak_device_memory_bytes": max(int(result["runtime_measurements"]["peak_device_memory_bytes"]) for result in all_results),
            "completed_epochs_total": sum(int(result["runtime_measurements"]["completed_epochs"]) for result in all_results),
            "completed_training_steps_total": sum(int(result["runtime_measurements"]["completed_training_steps"]) for result in all_results),
            "cache_input_volume_bytes": arrays.input_volume_bytes, "raw_cube_total_bytes_not_read": RAW_CUBE_TOTAL_BYTES,
            "frozen_limits_per_run": template.resource_budget.model_dump(mode="json"),
        },
        "scope": {
            "process_count": 1, "device": "cuda:0", "numeric_precision": "float32", "deterministic": True,
            "seeds": list(STRICT_SEEDS), "primary_threshold": 0.5,
            "nested_each_patient_exactly_once_per_seed": True, "lopo_each_patient_exactly_once_per_seed": True,
            "outer_labels_used_for_candidate_epoch_selection": False,
            "lopo_labels_used_for_candidate_epoch_selection": False,
            "hyperleaf_official_evaluation_started": False, "hyperleaf_official_evaluation_budget_consumed": 0,
            "read_only_inputs_unchanged": True,
        },
        "artifacts": {
            "initial_plan": "frozen-initial-plan.json", "final_plan": "frozen-final-plan.json",
            "outer_decisions": "frozen-outer-decisions.json", "lopo_policy": "frozen-lopo-policy.json",
            "secondary": "fitted-secondary-by-outer.json", "nested_protocol": "nested-protocol.canonical.json",
            "lopo_protocol": "lopo-protocol.canonical.json", "nested_root": "nested/", "lopo_root": "lopo/",
            "promotion": "promotion-evaluation.json", "negative_evidence": "negative-evidence.jsonl",
        },
        "caveats": [
            "This is complete cache-bound strict-protocol evidence, not publication-grade raw-lineage evidence.",
            "Historical local-token and spatial-mean cache bytes are content-hash bound but their creation lineage is incomplete.",
            "The registered historical local-token contract names full-cube spatial PCA; this report does not infer or claim that lineage from the cache bytes.",
            "Raw regeneration would require reading 622,309,721,600 bytes (579.57 GiB), beyond the frozen 8,000,000,000-byte input-volume budget.",
            "Primary metrics use threshold 0.5. Per-outer-fold fitted thresholds and temperatures are secondary and were frozen from inner OOF before outer labels were read.",
            "The task-15.12 global winner was validated only as a screening checkpoint and was not adopted as a nested-CV selection decision.",
        ],
    }
    _publish_json_once(execution_root / "task-15.13-report.json", report)
    latest = {
        "status": "complete_cache_bound", "execution_id": execution_id,
        "report": (execution_root / "task-15.13-report.json").relative_to(WORKSPACE).as_posix(),
        "promotion_status": claim.status.value, "raw_lineage_status": "blocked_input_volume_budget",
        "nested_evaluations_consumed": len(nested_results), "lopo_evaluations_consumed": len(lopo_results),
        "hyperleaf_official_evaluation_budget_consumed": 0,
    }
    _atomic_json(OUTPUT_ROOT / "latest-task-15.13-summary.json", latest)
    return latest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", metavar="EXECUTION_ID")
    args = parser.parse_args(argv)
    if args.execute == bool(args.resume):
        parser.error("choose exactly one of --execute or --resume EXECUTION_ID")
    result = execute_strict(resume_execution_id=args.resume)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
