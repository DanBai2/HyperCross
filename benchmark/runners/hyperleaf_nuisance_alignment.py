"""Execute task 25.1: pre-registered confirmatory strict LOVO over a nuisance-alignment arm family.

Why this exists
---------------
Task 24.4 already executed and fell back to the frozen 24.2 candidate in all four
outer folds.  Wave-9 showed that the governed inner-variety instrument is not the
reason: it detects the spectral variety-adversarial mechanism at +0.20 inner-OOF
Gsw.  The reason is that the frozen task-24.3 candidate space contained only
variants inside a single architecture (their inner-OOF Gsw spanned 0.0014) and
therefore contained no mechanism that could clear a promotion guard.

Task 25.1 therefore does not attempt inner-domain selection at all.  It
pre-registers a small, fixed arm family, trains every arm with one identical
predeclared budget, and evaluates every arm on the outer leave-one-variety-out
folds.  No outer metric is ever used to choose a configuration, so the outer
evaluation is confirmatory rather than selective.

Scope
-----
* G:/Datasets/HyperCross and G:/Datasets/HyperLeaf2024 only.
* Training partition only; official images, official labels and solution.csv stay sealed.
* Target normalization is fitted on the outer-train variety set only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.catalog.hyperleaf import HYPERLEAF_TARGETS, HYPERLEAF_VARIETIES
from benchmark.models.hyperleaf_candidates import (
    AVERAGE_BRANCH,
    PCA_BRANCH,
    PCA_INDEPENDENT_TARGET_MODEL_KEY,
    RGB_BRANCH,
    RGB_PCA_CONSISTENCY_MODEL_KEY,
    TARGET_ROUTED_BRANCH,
    rgb_pca_consistency_loss,
)
from benchmark.models.hyperleaf_nuisance_alignment import (
    SPECTRAL_ADVERSARIAL_03_MODEL_KEY,
    SPECTRAL_ADVERSARIAL_10_MODEL_KEY,
    SPECTRAL_ADVERSARIAL_BRANCH,
    SPECTRAL_CLEAN_BRANCH,
    SPECTRAL_ONLY_MODEL_KEY,
    SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY,
    create_hyperleaf_nuisance_alignment_model_registry,
)
from benchmark.remediation import load_remediation_template
from benchmark.runners.hyperleaf_nested_remediation import (
    OuterContext,
    build_nested_contexts,
    _load_nested_data,
    _publish_json_once,
    _read_json,
    _replace_adaptive_pool,
    _set_determinism,
    _utc_now,
    _validate_environment,
)
from benchmark.runners.hyperleaf_screening import (
    BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    WEIGHT_DECAY,
    _atomic_json,
    _load_train_manifest,
    _load_training_table,
    compute_grouped_metrics,
)
from benchmark.schema import FitBoundary, FitBoundaryScope, hash_file, hash_object
from benchmark.transforms.hyperleaf_candidates import (
    HYPERLEAF_STYLE_CONFIGS,
    apply_hyperleaf_style,
)
from benchmark.transforms.preprocessing import PreprocessingManager
from benchmark.protocols.hyperleaf_lovo import plan_hyperleaf_lovo
from benchmark.protocols.planner import ProtocolPlanner
from benchmark.recovery import FailureAtomicArtifactStore


WORKSPACE = Path(r"G:\Datasets\HyperCross")
DATA_ROOT = Path(r"G:\Datasets\HyperLeaf2024")
PROFILE_PATH = WORKSPACE / "configs" / "remediation" / "v1" / "deterministic-float32.json"
PREREGISTRATION_PATH = (
    WORKSPACE / "configs" / "optimization" / "v2" / "hyperleaf-task-25-preregistration.json"
)
OUTPUT_ROOT = WORKSPACE / "runs_v2" / "strict" / "hyperleaf-nuisance-alignment" / "v1" / "deterministic-float32"
SMOKE_ROOT = WORKSPACE / "runs_v2" / "strict" / "hyperleaf-nuisance-alignment" / "v1" / "smoke"
TASK_ID = "25.1"
SPACE_KEY = "hyperleaf.task-25.spectral-nuisance-alignment.v1"
SEEDS = (7, 42, 123)
FIXED_EPOCHS = 60
STYLE_KEY = "mild"
VARIETY_INDEX = {variety: index for index, variety in enumerate(HYPERLEAF_VARIETIES)}
SPECTRAL_ONLY_ROUTE = ("spectral-only", "spectral-only", "spectral-only")


@dataclass(frozen=True)
class Arm:
    """One pre-registered arm.  No arm is ever chosen using an outer metric."""

    arm_id: str
    role: str
    model_key: str
    target_routes: tuple[str, str, str]
    uses_rgb: bool
    uses_pca: bool
    adversarial: bool
    consistency_weight: float
    hypothesis: str

    def declaration(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "role": self.role,
            "model_key": self.model_key,
            "target_routes": list(self.target_routes),
            "uses_rgb": self.uses_rgb,
            "uses_pca": self.uses_pca,
            "adversarial": self.adversarial,
            "consistency_weight": self.consistency_weight,
            "hypothesis": self.hypothesis,
        }


ARMS: tuple[Arm, ...] = (
    Arm(
        arm_id="control::spectral-only::mse",
        role="matched_control",
        model_key=SPECTRAL_ONLY_MODEL_KEY,
        target_routes=SPECTRAL_ONLY_ROUTE,
        uses_rgb=False,
        uses_pca=False,
        adversarial=False,
        consistency_weight=0.0,
        hypothesis="Matched control: the same governed spectrum encoder without any domain alignment.",
    ),
    Arm(
        arm_id="arm::spectral-dann-lam0.3",
        role="optimization",
        model_key=SPECTRAL_ADVERSARIAL_03_MODEL_KEY,
        target_routes=SPECTRAL_ONLY_ROUTE,
        uses_rgb=False,
        uses_pca=False,
        adversarial=True,
        consistency_weight=0.0,
        hypothesis="Weak variety-adversarial alignment should move Gsw less than lambda 1.0 (monotonicity check).",
    ),
    Arm(
        arm_id="arm::spectral-dann-lam1.0",
        role="optimization",
        model_key=SPECTRAL_ADVERSARIAL_10_MODEL_KEY,
        target_routes=SPECTRAL_ONLY_ROUTE,
        uses_rgb=False,
        uses_pca=False,
        adversarial=True,
        consistency_weight=0.0,
        hypothesis="Full-strength spectral alignment should raise cross-variety Gsw R2 above the matched control.",
    ),
    Arm(
        arm_id="arm::spectral-routed-dann-lam1.0",
        role="optimization",
        model_key=SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY,
        target_routes=(SPECTRAL_CLEAN_BRANCH, SPECTRAL_ADVERSARIAL_BRANCH, SPECTRAL_CLEAN_BRANCH),
        uses_rgb=False,
        uses_pca=False,
        adversarial=True,
        consistency_weight=0.0,
        hypothesis="Routing Gsw to the aligned expert and GrainWeight/PhiPS2 to the clean expert should keep the Gsw gain without the GrainWeight loss.",
    ),
    Arm(
        arm_id="reference::rgb-pca-consistency-010",
        role="external_reference",
        model_key=RGB_PCA_CONSISTENCY_MODEL_KEY,
        target_routes=(RGB_BRANCH, AVERAGE_BRANCH, PCA_BRANCH),
        uses_rgb=True,
        uses_pca=True,
        adversarial=False,
        consistency_weight=0.10,
        hypothesis="In-run reference reproducing the frozen task-24.2 winning branch under the task-25 budget.",
    ),
    Arm(
        arm_id="reference::train-fold-pca",
        role="external_reference",
        model_key=PCA_INDEPENDENT_TARGET_MODEL_KEY,
        target_routes=(PCA_BRANCH, PCA_BRANCH, PCA_BRANCH),
        uses_rgb=False,
        uses_pca=True,
        adversarial=False,
        consistency_weight=0.0,
        hypothesis="In-run reference reproducing the frozen task-24.2 GrainWeight-best branch.",
    ),
)

PREREGISTERED_CRITERIA: tuple[dict[str, Any], ...] = (
    {
        "criterion_id": "C1",
        "statement": "spectral-dann-lam1.0 pooled-OOF Gsw R2 exceeds the matched spectral-only control by at least 0.03 in the seed mean and is strictly positive in every seed.",
        "metric": "pooled_oof.r2.Gsw",
        "arm": "arm::spectral-dann-lam1.0",
        "guard": "control::spectral-only::mse",
    },
    {
        "criterion_id": "C2",
        "statement": "spectral-routed-dann-lam1.0 keeps the Gsw gain (>= control + 0.03) while staying within 0.02 of the control on GrainWeight and PhiPS2.",
        "metric": "pooled_oof.r2",
        "arm": "arm::spectral-routed-dann-lam1.0",
        "guard": "control::spectral-only::mse",
    },
    {
        "criterion_id": "C3",
        "statement": "Gsw R2 is monotone in the adversarial strength: lam0.3 <= lam1.0.",
        "metric": "pooled_oof.r2.Gsw",
        "arms": ["arm::spectral-dann-lam0.3", "arm::spectral-dann-lam1.0"],
    },
    {
        "criterion_id": "C4",
        "statement": "The in-run RGB/PCA and train-fold-PCA references land in the task-24.2 range on Gsw, confirming harness comparability.",
        "metric": "pooled_oof.r2.Gsw",
        "arms": ["reference::rgb-pca-consistency-010", "reference::train-fold-pca"],
    },
)


def _arm_by_id(arm_id: str) -> Arm:
    for arm in ARMS:
        if arm.arm_id == arm_id:
            return arm
    raise ValueError(f"unknown task-25 arm: {arm_id!r}")

def _preregistration_payload() -> dict[str, Any]:
    """Return the frozen declaration written before any training happens."""

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "task_id": TASK_ID,
        "candidate_space_key": SPACE_KEY,
        "status": "frozen_before_outer_evaluation",
        "workspace": "G:/Datasets/HyperCross",
        "data_root": "G:/Datasets/HyperLeaf2024",
        "protocol": "leaf.lovo.nested.v1 outer folds, seeds 7/42/123",
        "training_policy": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": FIXED_EPOCHS,
            "epochs": FIXED_EPOCHS,
            "early_stopping": False,
            "epoch_selection": "not_applicable_fixed_budget",
            "style_augmentation": STYLE_KEY,
            "target_normalization": "outer-train-only",
            "adversarial_objective": "cross_entropy_on_variety_through_gradient_reversal",
        },
        "seals": {
            "official_images_allowed": False,
            "official_labels_allowed": False,
            "solution_csv_allowed": False,
            "outer_metrics_allowed_for_selection": False,
            "epoch_or_arm_selection_on_outer_metrics": False,
        },
        "arms": [arm.declaration() for arm in ARMS],
        "reference_identity": {
            "variant_candidate_space": "hyperleaf.task-24.3.gsw-target-routing.v1",
            "variant_task_24_4_execution_id": "task-24-4-20261028T055249572925Z",
            "variant_task_24_4_all_contexts_fell_back": True,
        },
        "pre_registered_criteria": list(PREREGISTERED_CRITERIA),
        "historical_design_basis": {
            "exploratory_arm_outer_gsw": {
                "clean_spectral_mlp_204_band": -0.1191,
                "spectral_variety_adversarial_lambda_1.0": 0.0523,
            },
            "selection_score_allowed": False,
            "outer_metric_revival_allowed": False,
        },
    }
    payload["preregistration_hash"] = hash_object(payload)
    return payload


def freeze_preregistration() -> dict[str, Any]:
    """Write the immutable declaration exactly once."""

    payload = _preregistration_payload()
    if PREREGISTRATION_PATH.exists():
        existing = _read_json(PREREGISTRATION_PATH)
        if existing != payload:
            raise RuntimeError(
                "task-25 preregistration already exists with different content; "
                "a frozen declaration must never be edited"
            )
        return existing
    _publish_json_once(PREREGISTRATION_PATH, payload)
    return payload


def _verify_preregistration() -> dict[str, Any]:
    if not PREREGISTRATION_PATH.exists():
        raise RuntimeError("task-25 preregistration is missing; run --freeze before --execute")
    frozen = _read_json(PREREGISTRATION_PATH)
    expected = _preregistration_payload()
    if frozen.get("preregistration_hash") != expected["preregistration_hash"]:
        raise RuntimeError("task-25 preregistration hash drift")
    if [arm["arm_id"] for arm in frozen["arms"]] != [arm.arm_id for arm in ARMS]:
        raise RuntimeError("task-25 arm declaration drift")
    return frozen


def _train_arm(
    arm: Arm,
    fold: Any,
    seed: int,
    data: Any,
    device: torch.device,
    epochs: int,
) -> Any:
    _set_determinism(seed)
    model = create_hyperleaf_nuisance_alignment_model_registry().build(arm.model_key)
    _replace_adaptive_pool(model)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_indices = np.asarray([data.id_to_index[sample_id] for sample_id in fold.train_ids], dtype=np.int64)
    artifact = data.target_artifacts[fold.fold_id]
    target_mean = np.asarray(artifact.state["mean"], dtype=np.float32)
    target_scale = np.asarray(artifact.state["scale"], dtype=np.float32)
    variety_codes = np.asarray([VARIETY_INDEX[variety] for variety in data.varieties], dtype=np.int64)
    style = HYPERLEAF_STYLE_CONFIGS[STYLE_KEY]
    pca_rows = data.pca_by_fold[fold.fold_id]

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(
            len(train_indices), generator=torch.Generator().manual_seed(seed * 100_000 + epoch)
        ).numpy()
        for batch_number, start in enumerate(range(0, len(order), BATCH_SIZE)):
            selected = train_indices[order[start:start + BATCH_SIZE]]
            rgb = torch.from_numpy(data.rgb[selected]).to(device=device, dtype=torch.float32)
            pca = torch.from_numpy(pca_rows[selected]).to(device=device, dtype=torch.float32)
            spectrum = torch.from_numpy(data.spectrum[selected]).to(device=device, dtype=torch.float32)
            targets = torch.from_numpy(
                ((data.targets[selected] - target_mean) / target_scale).astype(np.float32)
            ).to(device)
            variety_batch = torch.from_numpy(variety_codes[selected]).to(device)
            augmentation_seed = seed * 10_000_000 + epoch * 10_000 + batch_number
            rgb, styled_spectrum = apply_hyperleaf_style(rgb, spectrum, config=style, seed=augmentation_seed)
            pca, _ = apply_hyperleaf_style(pca, spectrum, config=style, seed=augmentation_seed)

            optimizer.zero_grad(set_to_none=True)
            loss = _arm_loss(model, arm, rgb, pca, styled_spectrum, targets, variety_batch)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite task-25 loss: {arm.arm_id}/{fold.fold_id}/{seed}/{epoch}")
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite task-25 gradient: {arm.arm_id}/{fold.fold_id}/{seed}/{epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


# --------------------------------------------------------------- arm objective ---
def _arm_directory(arm: Arm) -> str:
    return "".join(
        character if (character.isalnum() or character in "-_") else "-" for character in arm.arm_id
    )


def _predict_arm(
    model: Any,
    arm: Arm,
    rgb: torch.Tensor,
    pca: torch.Tensor,
    spectrum: torch.Tensor,
) -> torch.Tensor:
    """Return normalized three-target predictions for one declared arm."""

    with torch.no_grad():
        if arm.model_key == RGB_PCA_CONSISTENCY_MODEL_KEY:
            return model(
                rgb,
                pca,
                spectrum,
                inference_branch=TARGET_ROUTED_BRANCH,
                target_routes=arm.target_routes,
            )
        if arm.model_key == PCA_INDEPENDENT_TARGET_MODEL_KEY:
            return model(pca, spectrum)
        if arm.model_key == SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY:
            return model(spectrum, target_routes=arm.target_routes)
        return model(spectrum)


def _arm_loss(
    model: Any,
    arm: Arm,
    rgb: torch.Tensor,
    pca: torch.Tensor,
    spectrum: torch.Tensor,
    targets: torch.Tensor,
    variety_batch: torch.Tensor,
) -> torch.Tensor:
    """Return the pre-declared objective for one arm.

    Every arm minimizes plain mean-squared error over the outer-train-standardized
    three-target vector.  The spectral adversarial arms add a wheat-variety
    cross-entropy behind the gradient reversal; the RGB/PCA reference adds the
    frozen prediction-consistency penalty on its two branches.
    """

    if arm.model_key == RGB_PCA_CONSISTENCY_MODEL_KEY:
        _, branches = model(
            rgb,
            pca,
            spectrum,
            inference_branch=TARGET_ROUTED_BRANCH,
            target_routes=arm.target_routes,
            return_branches=True,
        )
        branch_loss = 0.5 * (
            F.mse_loss(branches[RGB_BRANCH], targets) + F.mse_loss(branches[PCA_BRANCH], targets)
        )
        penalty = rgb_pca_consistency_loss(
            branches[RGB_BRANCH], branches[PCA_BRANCH], weight=arm.consistency_weight
        )
        return branch_loss + penalty
    if arm.model_key == PCA_INDEPENDENT_TARGET_MODEL_KEY:
        return F.mse_loss(model(pca, spectrum), targets)
    if arm.model_key == SPECTRAL_ROUTED_ADVERSARIAL_MODEL_KEY:
        task = F.mse_loss(model(spectrum, target_routes=arm.target_routes), targets)
        return task + F.cross_entropy(model.variety_logits(spectrum), variety_batch)
    if arm.model_key == SPECTRAL_ONLY_MODEL_KEY:
        return F.mse_loss(model(spectrum), targets)
    if arm.model_key in (SPECTRAL_ADVERSARIAL_03_MODEL_KEY, SPECTRAL_ADVERSARIAL_10_MODEL_KEY):
        task = F.mse_loss(model(spectrum), targets)
        return task + F.cross_entropy(model.variety_logits(spectrum), variety_batch)
    raise ValueError(f"task-25 arm has no declared objective: {arm.arm_id!r}")


def _evaluate_arm(
    arm: Arm,
    data: Any,
    contexts: Sequence[OuterContext],
    seed: int,
    device: torch.device,
    epochs: int,
) -> dict[str, Any]:
    """Train one arm on every outer fold and return pooled out-of-fold evidence."""

    sample_ids: list[str] = []
    fold_ids: list[str] = []
    truth_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    for context in contexts:
        fold = context.fold
        model = _train_arm(arm, fold, seed, data, device, epochs)
        artifact = data.target_artifacts[fold.fold_id]
        mean = np.asarray(artifact.state["mean"], dtype=np.float64)
        scale = np.asarray(artifact.state["scale"], dtype=np.float64)
        indices = np.asarray(
            [data.id_to_index[sample_id] for sample_id in fold.validation_ids], dtype=np.int64
        )
        outputs: list[np.ndarray] = []
        model.eval()
        for start in range(0, len(indices), EVALUATION_BATCH_SIZE):
            selected = indices[start : start + EVALUATION_BATCH_SIZE]
            rgb = torch.from_numpy(data.rgb[selected]).to(device=device, dtype=torch.float32)
            pca = torch.from_numpy(data.pca_by_fold[fold.fold_id][selected]).to(
                device=device, dtype=torch.float32
            )
            spectrum = torch.from_numpy(data.spectrum[selected]).to(device=device, dtype=torch.float32)
            outputs.append(_predict_arm(model, arm, rgb, pca, spectrum).cpu().numpy().astype(np.float64))
        predicted = np.concatenate(outputs) * scale + mean
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"non-finite task-25 prediction: {arm.arm_id}/{fold.fold_id}/{seed}")
        sample_ids.extend(fold.validation_ids)
        fold_ids.extend([fold.fold_id] * len(fold.validation_ids))
        truth_rows.append(np.asarray([data.targets[index] for index in indices], dtype=np.float64))
        prediction_rows.append(predicted)
        del model
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(f"task-25 outer OOF coverage repeats a sample: {arm.arm_id}/{seed}")
    truth = np.concatenate(truth_rows)
    prediction = np.concatenate(prediction_rows)
    metrics = compute_grouped_metrics(tuple(sample_ids), tuple(fold_ids), truth, prediction)
    return {
        "metrics": metrics,
        "sample_ids": tuple(sample_ids),
        "fold_ids": tuple(fold_ids),
        "y_true": truth,
        "y_pred": prediction,
    }


def _seed_aggregate(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate pooled out-of-fold R2 across the pre-registered seeds."""

    ordered = sorted(per_seed)
    aggregate: dict[str, Any] = {}
    for target in HYPERLEAF_TARGETS:
        values = np.asarray(
            [per_seed[seed]["pooled_oof"]["targets"][target]["r2"] for seed in ordered],
            dtype=np.float64,
        )
        aggregate[target] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "per_seed": {str(seed): float(value) for seed, value in zip(ordered, values)},
        }
    return aggregate


# ---------------------------------------------------------------- criteria ---
def _criterion_verdicts(
    aggregate: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    """Evaluate the four pre-registered criteria against the frozen thresholds."""

    control = aggregate["control::spectral-only::mse"]
    dann_03 = aggregate["arm::spectral-dann-lam0.3"]
    dann_10 = aggregate["arm::spectral-dann-lam1.0"]
    routed = aggregate["arm::spectral-routed-dann-lam1.0"]
    gsw_delta = {
        str(seed): dann_10["Gsw"]["per_seed"][str(seed)] - control["Gsw"]["per_seed"][str(seed)]
        for seed in sorted(seeds)
    }
    c1 = bool(
        dann_10["Gsw"]["mean"] - control["Gsw"]["mean"] >= 0.03
        and all(value > 0.0 for value in gsw_delta.values())
    )
    routed_gsw = routed["Gsw"]["mean"] - control["Gsw"]["mean"]
    routed_gw = abs(routed["GrainWeight"]["mean"] - control["GrainWeight"]["mean"])
    routed_phi = abs(routed["PhiPS2"]["mean"] - control["PhiPS2"]["mean"])
    c2 = bool(routed_gsw >= 0.03 and routed_gw <= 0.02 and routed_phi <= 0.02)
    c3 = bool(dann_03["Gsw"]["mean"] <= dann_10["Gsw"]["mean"])
    reference_ids = ("reference::rgb-pca-consistency-010", "reference::train-fold-pca")
    c4 = bool(all(0.0 <= aggregate[arm_id]["Gsw"]["mean"] <= 0.20 for arm_id in reference_ids))
    return {
        "C1": {
            "met": c1,
            "gsw_delta_mean": float(dann_10["Gsw"]["mean"] - control["Gsw"]["mean"]),
            "gsw_delta_per_seed": {key: float(value) for key, value in gsw_delta.items()},
        },
        "C2": {
            "met": c2,
            "gsw_delta_mean": float(routed_gsw),
            "grainweight_gap": float(routed_gw),
            "phips2_gap": float(routed_phi),
        },
        "C3": {
            "met": c3,
            "lam0.3_gsw_mean": float(dann_03["Gsw"]["mean"]),
            "lam1.0_gsw_mean": float(dann_10["Gsw"]["mean"]),
        },
        "C4": {
            "met": c4,
            "reference_gsw_means": {arm_id: float(aggregate[arm_id]["Gsw"]["mean"]) for arm_id in reference_ids},
        },
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Task 25.1 - spectral nuisance-alignment arm family (development tier)")
    lines.append("")
    lines.append(f"- execution_id: `{payload['execution_id']}`")
    lines.append(f"- preregistration hash: `{payload['preregistration_hash']}`")
    lines.append(f"- protocol: `{payload['protocol_key']}` hash `{payload['protocol_hash']}`")
    lines.append(f"- budget: {payload['epochs']} epochs / seeds {payload['seeds']} / device `{payload['device']}`")
    lines.append(f"- claim status: `{payload['claim_and_promotion']['claim_status']}`")
    lines.append(f"- official evaluation budget consumed: {payload['official_evaluation_budget_consumed']}")
    lines.append("")
    lines.append("## Pooled out-of-fold R2 (mean +/- std over seeds)")
    lines.append("")
    lines.append("| arm | role | GrainWeight | Gsw | PhiPS2 |")
    lines.append("|---|---|---:|---:|---:|")
    for arm in ARMS:
        summary = payload["arm_summaries"].get(arm.arm_id)
        if summary is None:
            continue
        aggregate = summary["aggregate"]
        cells = [
            f"{aggregate[target]['mean']:.4f} +/- {aggregate[target]['std']:.4f}"
            for target in HYPERLEAF_TARGETS
        ]
        lines.append(f"| `{arm.arm_id}` | {arm.role} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Pre-registered criteria")
    lines.append("")
    lines.append("| criterion | met | evidence |")
    lines.append("|---|---|---|")
    for criterion in PREREGISTERED_CRITERIA:
        criterion_id = str(criterion["criterion_id"])
        verdict = payload["criterion_verdicts"].get(criterion_id)
        if verdict is None:
            lines.append(f"| {criterion_id} | n/a | not evaluated in this partial run |")
            continue
        evidence = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in verdict.items()
            if key != "met"
        )
        lines.append(f"| {criterion_id} | {'YES' if verdict['met'] else 'NO'} | {evidence} |")
    lines.append("")
    lines.append("## Declared criteria text")
    lines.append("")
    for criterion in PREREGISTERED_CRITERIA:
        lines.append(f"- **{criterion['criterion_id']}**: {criterion['statement']}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------- execute ---
def execute(
    *,
    resume_execution_id: str | None = None,
    epochs: int = FIXED_EPOCHS,
    seeds: Sequence[int] = SEEDS,
    arm_ids: Sequence[str] | None = None,
    smoke: bool = False,
    fold_limit: int = 0,
) -> dict[str, Any]:
    """Run the pre-registered arm family on the outer leave-one-variety-out folds."""

    for root in (WORKSPACE, DATA_ROOT):
        if PureWindowsPath(str(root)).drive.casefold() != "g:":
            raise RuntimeError("task 25.1 is restricted to approved G: roots")
    frozen = _verify_preregistration()
    template = load_remediation_template(PROFILE_PATH)
    environment = _validate_environment(template)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    manifest = _load_train_manifest()
    sample_ids, _, targets = _load_training_table(manifest)
    planner = ProtocolPlanner(FailureAtomicArtifactStore(OUTPUT_ROOT))
    protocol = plan_hyperleaf_lovo(planner, manifest, seed_set=SEEDS, publish=False)
    contexts = build_nested_contexts(protocol, manifest)
    if fold_limit:
        contexts = contexts[: int(fold_limit)]

    selected_arms = tuple(
        _arm_by_id(arm_id) for arm_id in (arm_ids or tuple(arm.arm_id for arm in ARMS))
    )
    output_root = SMOKE_ROOT if smoke else OUTPUT_ROOT
    execution_id = resume_execution_id or datetime.now(timezone.utc).strftime("task-25-1-%Y%m%dT%H%M%S%fZ")
    execution_root = output_root / execution_id
    execution_root.mkdir(parents=True, exist_ok=True)

    started_ns = time.monotonic_ns()
    data = _load_nested_data(
        manifest, sample_ids, contexts, targets, protocol.protocol_hash, execution_root
    )

    arm_summaries: dict[str, Any] = {}
    for arm in selected_arms:
        per_seed: dict[int, Any] = {}
        result_hashes: dict[str, str] = {}
        for seed in seeds:
            result_path = execution_root / "arms" / _arm_directory(arm) / f"seed-{int(seed)}.json"
            if result_path.exists():
                result = _read_json(result_path)
            else:
                evaluation = _evaluate_arm(arm, data, contexts, int(seed), device, int(epochs))
                result = {
                    "schema_version": "1.0",
                    "task_id": TASK_ID,
                    "arm_id": arm.arm_id,
                    "model_key": arm.model_key,
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "held_out_varieties": [context.fold.validation_variety for context in contexts],
                    "sample_count": len(evaluation["sample_ids"]),
                    "metrics": evaluation["metrics"],
                    "records": [
                        {
                            "sample_id": sample_id,
                            "fold_id": fold_id,
                            "y_true": [float(value) for value in truth_row],
                            "y_pred": [float(value) for value in predicted_row],
                        }
                        for sample_id, fold_id, truth_row, predicted_row in zip(
                            evaluation["sample_ids"],
                            evaluation["fold_ids"],
                            evaluation["y_true"],
                            evaluation["y_pred"],
                        )
                    ],
                }
                result["result_hash"] = hash_object(result)
                _publish_json_once(result_path, result)
            per_seed[int(seed)] = result["metrics"]
            result_hashes[str(int(seed))] = str(result["result_hash"])
            print(
                json.dumps(
                    {
                        "stage": "task_25_1_arm_seed_complete",
                        "arm_id": arm.arm_id,
                        "seed": int(seed),
                        "pooled_gsw_r2": result["metrics"]["pooled_oof"]["targets"]["Gsw"]["r2"],
                    }
                ),
                flush=True,
            )
        arm_summaries[arm.arm_id] = {
            "role": arm.role,
            "declaration": arm.declaration(),
            "per_seed_metrics": {str(seed): per_seed[seed] for seed in sorted(per_seed)},
            "aggregate": _seed_aggregate(per_seed),
            "result_hashes": result_hashes,
        }

    criterion_verdicts: dict[str, Any] = {}
    if all(arm.arm_id in arm_summaries for arm in ARMS):
        criterion_verdicts = _criterion_verdicts(
            {arm_id: summary["aggregate"] for arm_id, summary in arm_summaries.items()}, seeds
        )

    report = {
        "schema_version": "1.0",
        "task_id": TASK_ID,
        "candidate_space_key": SPACE_KEY,
        "execution_id": execution_id,
        "created_utc": _utc_now(),
        "preregistration_path": PREREGISTRATION_PATH.relative_to(WORKSPACE).as_posix(),
        "preregistration_hash": frozen["preregistration_hash"],
        "protocol_key": protocol.protocol_key,
        "protocol_hash": protocol.protocol_hash,
        "dataset_hash": manifest.dataset_hash,
        "epochs": int(epochs),
        "seeds": [int(seed) for seed in seeds],
        "smoke": bool(smoke),
        "fold_limit": int(fold_limit),
        "outer_folds": [
            {
                "fold_id": context.fold.fold_id,
                "held_out_variety": context.fold.validation_variety,
                "validation_count": len(context.fold.validation_ids),
            }
            for context in contexts
        ],
        "device": torch.cuda.get_device_name(0),
        "environment_lock": environment.model_dump(mode="json"),
        "arm_summaries": arm_summaries,
        "criterion_verdicts": criterion_verdicts,
        "official_evaluation_budget_consumed": 0,
        "wall_clock_seconds": (time.monotonic_ns() - started_ns) / 1e9,
        "claim_and_promotion": {
            "claim_status": "development",
            "promotion_permitted": False,
            "outer_metrics_used_for_selection": False,
            "rationale": (
                "Task 25.1 evaluates a pre-registered, fixed-budget arm family on the outer "
                "leave-one-variety-out folds and never uses an outer metric to choose a configuration or "
                "epoch.  It is development tier rather than strict_candidate because it does not pass "
                "through the governed inner-variety selection pipeline of the task 24.x stack; a positive "
                "verdict here is the prerequisite for a superseding governed entry, not a claim."
            ),
        },
    }
    _atomic_json(execution_root / "task-25.1-report.json", report)
    _write_markdown(execution_root / "task-25.1-summary.md", report)
    print(
        json.dumps(
            {
                "stage": "task_25_1_complete",
                "execution_id": execution_id,
                "report": (execution_root / "task-25.1-report.json").relative_to(WORKSPACE).as_posix(),
                "summary": (execution_root / "task-25.1-summary.md").relative_to(WORKSPACE).as_posix(),
                "criterion_verdicts": {
                    key: value["met"] for key, value in criterion_verdicts.items()
                },
            }
        ),
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task 25.1 pre-registered spectral nuisance-alignment LOVO evaluation"
    )
    parser.add_argument("--freeze", action="store_true", help="write the immutable pre-registration and exit")
    parser.add_argument("--execute", action="store_true", help="execute the pre-registered arm family")
    parser.add_argument("--resume", default=None, help="resume an existing execution id")
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--arms", default=",".join(arm.arm_id for arm in ARMS))
    parser.add_argument("--smoke", action="store_true", help="write into the smoke root")
    parser.add_argument("--fold-limit", type=int, default=0, help="restrict to the first N outer folds")
    arguments = parser.parse_args(argv)

    if arguments.freeze:
        payload = freeze_preregistration()
        print(json.dumps({
            "status": "frozen",
            "path": PREREGISTRATION_PATH.relative_to(WORKSPACE).as_posix(),
            "preregistration_hash": payload["preregistration_hash"],
        }, indent=2))
        return 0
    if not arguments.execute and arguments.resume is None:
        parser.error("one of --freeze, --execute or --resume is required")

    seeds = tuple(int(value) for value in arguments.seeds.split(",") if value.strip())
    arm_ids = tuple(value for value in arguments.arms.split(",") if value.strip())
    report = execute(
        resume_execution_id=arguments.resume,
        epochs=int(arguments.epochs),
        seeds=seeds,
        arm_ids=arm_ids,
        smoke=bool(arguments.smoke),
        fold_limit=int(arguments.fold_limit),
    )
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
