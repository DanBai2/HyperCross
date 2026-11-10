"""Append-only selected-policy projections for a completed task-15.13 run."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from benchmark.runners.histology_strict import (
    OUTPUT_ROOT,
    STRICT_SEEDS,
    WORKSPACE,
    _atomic_json,
    _load_rows,
    _metric_payload,
    _read_json,
    _safe_id,
)
from benchmark.schema import hash_file, hash_object


def _fold_macro(per_fold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity", "ece"):
        values = [payload[metric]["value"] for payload in per_fold.values() if payload[metric]["value"] is not None]
        result[metric] = {
            "value": None if not values else float(np.mean(values)),
            "defined_count": len(values),
            "total_count": len(per_fold),
        }
    return result


def _aggregate_seed_values(per_seed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in ("pooled_oof", "fold_macro"):
        result[scope] = {}
        for metric in ("accuracy", "balanced_accuracy", "auroc", "sensitivity", "specificity", "ece"):
            values = []
            for seed in STRICT_SEEDS:
                payload = per_seed[str(seed)][scope][metric]
                value = payload["value"]
                if value is not None:
                    values.append(float(value))
            result[scope][metric] = {
                "seed_mean": None if not values else float(np.mean(values)),
                "seed_variance": None if len(values) < 2 else float(np.var(values, ddof=1)),
                "defined_seed_count": len(values),
            }
    return result


def _project_protocol(execution_root: Path, protocol: str, selection_by_fold: Mapping[str, str], group_by_sample: Mapping[str, str]) -> dict[str, Any]:
    result_by_key = {}
    for path in (execution_root / protocol / "jobs").rglob("result.json"):
        result = _read_json(path)
        result_by_key[(str(result["candidate_id"]), int(result["seed"]), str(result["fold_id"]))] = result
    per_seed: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    expected_ids = tuple(sorted(group_by_sample, key=str.encode))
    for seed in STRICT_SEEDS:
        rows = []
        member_run_ids = []
        prediction_hashes = []
        for fold_id, candidate_id in sorted(selection_by_fold.items()):
            result = result_by_key[(candidate_id, seed, fold_id)]
            rows.extend(_load_rows(execution_root, result))
            member_run_ids.append(result["run_id"])
            prediction_hashes.append(result["prediction_content_hash"])
        if len(rows) != len(expected_ids) or {row["sample_id"] for row in rows} != set(expected_ids):
            raise RuntimeError(f"{protocol} seed {seed} selected-policy OOF coverage mismatch")
        metric_rows = [{**row, "inference_branch": f"{protocol}.selected-policy"} for row in rows]
        pooled = _metric_payload(metric_rows, f"{protocol}.selected-policy.seed-{seed}.pooled")
        per_fold = {
            fold_id: _metric_payload(
                [row for row in rows if row["fold_id"] == fold_id],
                f"{protocol}.selected-policy.seed-{seed}.{fold_id}",
            )
            for fold_id in sorted(selection_by_fold, key=str.encode)
        }
        per_patient = {
            patient: _metric_payload(
                [row for row in rows if row["group_id"] == patient],
                f"{protocol}.selected-policy.seed-{seed}.patient-{patient}",
            )
            for patient in sorted(set(group_by_sample.values()), key=str.encode)
        }
        metrics = {"pooled_oof": pooled, "fold_macro": _fold_macro(per_fold), "per_fold": per_fold, "per_patient": per_patient}
        payload = {
            "schema_version": "1.0", "task_id": "15.13", "protocol": protocol,
            "policy": "per-outer-fold-selected" if protocol == "nested" else "nested-inner-frozen-fixed-epoch-lopo",
            "seed": seed, "member_run_ids": member_run_ids, "prediction_hashes": prediction_hashes,
            "sample_count": len(rows), "metrics": metrics, "records_extended": rows,
        }
        payload["aggregate_hash"] = hash_object(payload)
        path = execution_root / protocol / "aggregates" / "selected-policy" / f"seed-{seed}-oof.json"
        if path.exists():
            existing = _read_json(path)
            if existing != payload:
                raise RuntimeError(f"immutable selected-policy artifact differs: {path}")
        else:
            _atomic_json(path, payload)
        artifacts[str(seed)] = {"path": path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(path)}
        per_seed[str(seed)] = metrics
    return {"per_seed_metrics": per_seed, "aggregate_metrics": _aggregate_seed_values(per_seed), "seed_oof_artifacts": artifacts}


def finalize(execution_id: str) -> dict[str, Any]:
    execution_root = OUTPUT_ROOT / "executions" / execution_id
    report_path = execution_root / "task-15.13-report.json"
    report = _read_json(report_path)
    if report.get("status") != "complete_cache_bound":
        raise RuntimeError("task-15.13 main report is not complete")
    manifest = _read_json(WORKSPACE / "artifacts" / "manifests" / "histology-hsi-gb" / "ba6a973ffa80636d42d50e274719bb72b4907766056b63e90d078da472704e12.json")
    group_by_sample = {str(record["sample_id"]): str(record["group_id"]) for record in manifest["records"]}
    nested_selection = {fold_id: str(value["selected_candidate"]) for fold_id, value in report["outer_fold_decisions"].items()}
    lopo_candidate = str(report["lopo_policy"]["selected_candidate"])
    lopo_folds = sorted({path.parent.name for path in (execution_root / "lopo" / "jobs" / _safe_id(lopo_candidate) / "seed-7").glob("*/result.json")}, key=str.encode)
    lopo_selection = {fold_id: lopo_candidate for fold_id in lopo_folds}
    supplement = {
        "schema_version": "1.0", "task_id": "15.13", "execution_id": execution_id,
        "append_only_supplement_to": report_path.relative_to(WORKSPACE).as_posix(),
        "source_report_hash": hash_file(report_path),
        "nested_selected_policy": _project_protocol(execution_root, "nested", nested_selection, group_by_sample),
        "lopo_selected_policy": _project_protocol(execution_root, "lopo", lopo_selection, group_by_sample),
        "secondary_policy": {
            "primary_label": "primary.fixed-threshold-0.5",
            "fitted_threshold_label": "secondary.inner-oof-fitted-threshold",
            "fitted_temperature_label": "secondary.inner-oof-fitted-temperature",
            "nested_values_by_outer_fold": {
                fold_id: {
                    "threshold": value["secondary"]["fitted_threshold"]["value"],
                    "temperature": value["secondary"]["fitted_temperature"]["value"],
                    "source_prediction_hash": value["secondary"]["source_prediction_hash"],
                }
                for fold_id, value in report["outer_fold_decisions"].items()
            },
        },
        "candidate_control_table_reference": "task-15.13-report.json#/nested/candidate_control_table and #/lopo/candidate_control_table",
        "patient_uncertainty_reference": "task-15.13-report.json#/nested/selected_policy/patient_bootstrap_10000 and #/lopo/selected_policy/patient_bootstrap_10000",
        "claim_status": report["promotion_evaluation"]["status"],
    }
    supplement["supplement_hash"] = hash_object(supplement)
    path = execution_root / "task-15.13-report-supplement.json"
    if path.exists():
        if _read_json(path) != supplement:
            raise RuntimeError("immutable supplement differs")
    else:
        _atomic_json(path, supplement)
    return {"status": "complete", "supplement": path.relative_to(WORKSPACE).as_posix(), "hash": hash_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_id")
    args = parser.parse_args()
    print(finalize(args.execution_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
