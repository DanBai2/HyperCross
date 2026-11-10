"""Read-only compatibility audit for legacy G-drive PyTorch checkpoints.

The audit hash-gates every checkpoint before restricted deserialization, matches
its state dictionary against known legacy model classes, strictly reloads the
weights, validates one real cached input batch against the inferred input
contract, and runs one bounded CUDA forward pass. Legacy files are never opened
for writing. Results are published only below ``runs_v2/audit``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from benchmark.models.checkpoint import ExternalCheckpointLoadError, ExternalCheckpointLoader

_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
_ALLOWED_DRIVE = "G:"


@dataclass(frozen=True)
class ModelCandidate:
    """One known legacy model construction and its expected input contract."""

    name: str
    factory: Callable[[], Any]
    output_dim: int
    spectrum_kind: str
    visual_channels: int
    source_ref: str


@dataclass(frozen=True)
class InputFixture:
    """One read-only cached input batch and its inferred contract."""

    visual: Any
    spectrum: Any
    source_ref: str
    representation_key: str
    tensor_shapes: Mapping[str, tuple[int, ...]]
    dtypes: Mapping[str, str]
    channel_or_band_order: Mapping[str, Sequence[str]]


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def state_signature(state: Mapping[str, Any]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Return the exact ordered key/shape/dtype signature of a tensor state dict."""

    result: list[tuple[str, tuple[int, ...], str]] = []
    for key, value in state.items():
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"state entry {key!r} is not tensor-like")
        result.append((str(key), tuple(int(x) for x in value.shape), str(value.dtype)))
    return tuple(result)


def extract_model_state(payload: Any) -> Mapping[str, Any]:
    """Extract a tensor state dictionary from a restricted-load payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload is not a mapping")
    state = payload.get("model", payload)
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint has no non-empty model state dictionary")
    state_signature(state)
    return state


def _factory(module: Any, class_name: str, *args: Any, **kwargs: Any) -> Callable[[], Any]:
    return lambda: getattr(module, class_name)(*args, **kwargs)


def build_candidates(src_root: Path) -> tuple[list[ModelCandidate], list[dict[str, str]]]:
    """Import known read-only legacy model definitions and build candidates."""

    src_text = str(src_root)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

    unavailable: list[dict[str, str]] = []
    modules: dict[str, Any] = {}
    for module_name in (
        "hypercross_routeA",
        "modern_hsi_models",
        "run_globalv_local_spectral",
        "run_local_mamtrans",
        "run_masked_pretrain",
        "run_hist_patient_cv_confidence",
        "run_hist_patient_cv_domainadv",
        "run_hist_patient_cv_invariant",
        "run_official_spectralformer",
    ):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except Exception as exc:  # missing optional upstream dependency is evidence
            unavailable.append(
                {
                    "module": module_name,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )

    candidates: list[ModelCandidate] = []

    def add(
        module_name: str,
        class_name: str,
        output_dim: int,
        spectrum_kind: str = "global_128",
        visual_channels: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        module = modules.get(module_name)
        if module is None:
            return
        candidates.append(
            ModelCandidate(
                name=f"{module_name}.{class_name}[out_dim={output_dim}]",
                factory=_factory(module, class_name, *args, **kwargs),
                output_dim=output_dim,
                spectrum_kind=spectrum_kind,
                visual_channels=visual_channels,
                source_ref=f"src/{module_name}.py",
            )
        )

    h = "hypercross_routeA"
    for output_dim in (2, 3):
        task = "cls" if output_dim == 2 else "reg"
        add(h, "VisualOnly", output_dim, "global_128", 3, output_dim)
        add(h, "SpectralOnly", output_dim, "global_128", 3, output_dim)
        add(h, "ConcatModel", output_dim, "global_128", 3, output_dim)
        for class_name in ("CrossModalRouteA", "TokenRouteA", "CorrespondenceRouteA", "HybridRouteA", "BilinearRouteA", "ResidualRouteA", "StatsFusionRouteA", "StatsResidualRouteA", "GroupNormCrossModalRouteA", "MultiHeadCrossModal"):
            add(h, class_name, output_dim, "global_128", 3, task, output_dim)
        add(h, "DualVisualCrossModalRouteA", output_dim, "global_128", 6, task, output_dim)
        for class_name in ("MILRouteA", "MILConcat", "MILLocalRouteA", "MILAlignedRouteA"):
            add(h, class_name, output_dim, "local_16x128", 3, output_dim)
        add(h, "SpectralTransformer", output_dim, "global_128", 3, output_dim)
        add(h, "TinyViT", output_dim, "global_128", 3, output_dim)
        add(h, "FiLMFusion", output_dim, "global_128", 3, output_dim)

        add("modern_hsi_models", "HybridSNLite", output_dim, "global_128", 3, output_dim)
        add("modern_hsi_models", "S2MambaLite", output_dim, "global_128", 3, output_dim)
        add("modern_hsi_models", "MamTransLite", output_dim, "global_128", 3, output_dim)
        add("run_globalv_local_spectral", "GlobalVLocalSpectral", output_dim, "local_16x128", 3, task, output_dim)
        add("run_local_mamtrans", "LocalMamTrans", output_dim, "local_16x128", 3, task, output_dim)
        add("run_masked_pretrain", "MaskedMILRouteA", output_dim, "local_16x128", 3, output_dim)
        add("run_official_spectralformer", "OfficialSpectralFormer", output_dim, "global_128", 3, task, output_dim)

    add("run_hist_patient_cv_confidence", "ConfidenceMILRouteA", 2, "local_16x128", 3, 2)
    add("run_hist_patient_cv_invariant", "InvariantMILRouteA", 2, "local_16x128", 3, 2)
    for domains in (9, 10, 11):
        add("run_hist_patient_cv_domainadv", "DomainAdvMILRouteA", 2, "local_16x128", 3, 2, n_domains=domains)

    return candidates, unavailable


def build_signature_index(candidates: Iterable[ModelCandidate]) -> tuple[dict[tuple[Any, ...], list[ModelCandidate]], list[dict[str, str]]]:
    """Index candidates by exact state key, shape, and dtype signature."""

    index: dict[tuple[Any, ...], list[ModelCandidate]] = {}
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            model = candidate.factory()
            signature = state_signature(model.state_dict())
            index.setdefault(signature, []).append(candidate)
        except Exception as exc:
            failures.append(
                {
                    "candidate": candidate.name,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                }
            )
    return index, failures


def _candidate_hint_score(candidate: ModelCandidate, checkpoint_ref: str) -> tuple[int, str]:
    text = checkpoint_ref.lower().replace("_", "")
    class_name = candidate.name.rsplit(".", 1)[-1].split("[", 1)[0].lower()
    aliases = {
        "crossmodalroutea": ("routea",),
        "tokenroutea": ("routeatoken",),
        "hybridroutea": ("routeahybrid",),
        "bilinearroutea": ("routeabilinear",),
        "residualroutea": ("routearesidual",),
        "milroutea": ("routeamil", "routeaalt", "routeafinal", "routeabest", "best.pt"),
        "millocalroutea": ("routeamillocal",),
        "milconcat": ("milconcat",),
        "mamtranslite": ("mamtranslite", "mamtransalt"),
        "officialspectralformer": ("spectralformerofficialcaf",),
    }
    tokens = (class_name, *aliases.get(class_name, ()))
    score = max((len(token) for token in tokens if token in text), default=0)
    return (-score, candidate.name)


def choose_candidate(matches: Sequence[ModelCandidate], checkpoint_ref: str) -> tuple[ModelCandidate | None, list[str]]:
    """Choose deterministically among structurally identical model candidates."""

    if not matches:
        return None, []
    ranked = sorted(matches, key=lambda item: _candidate_hint_score(item, checkpoint_ref))
    return ranked[0], [item.name for item in ranked]


def infer_dataset_key(checkpoint_ref: str, output_dim: int) -> str:
    text = checkpoint_ref.lower().replace("\\", "/")
    if "/leaf/" in text or text.startswith("runs/leaf_") or "leaf_" in Path(text).name:
        return "hyperleaf2024"
    if "/hist/" in text or text.startswith("runs/hist_") or "hist_" in text:
        return "histology-hsi-gb"
    return "histology-hsi-gb" if output_dim == 2 else "hyperleaf2024"


def _first_npz(cache_dir: Path) -> Path:
    try:
        return next(path for path in sorted(cache_dir.glob("*.npz")) if path.is_file())
    except StopIteration as exc:
        raise FileNotFoundError(f"no cached input found in {cache_dir}") from exc


def load_input_fixture(
    candidate: ModelCandidate,
    dataset_key: str,
    dataset_roots: Mapping[str, Path],
) -> InputFixture:
    """Load one real, read-only cached batch for the inferred contract."""

    import torch

    root = dataset_roots[dataset_key]
    local = candidate.spectrum_kind == "local_16x128"
    cache_name = (
        "cache_routeA_leaf_mil" if dataset_key == "hyperleaf2024" and local
        else "cache_routeA_leaf" if dataset_key == "hyperleaf2024"
        else "cache_routeA_hist_mil" if local
        else "cache_routeA_v2"
    )
    cache_file = _first_npz(root / cache_name)
    with np.load(cache_file, allow_pickle=False) as payload:
        visual_array = np.asarray(payload["v"], dtype=np.float32)
        spectrum_array = np.asarray(payload["s"], dtype=np.float32)

    if visual_array.ndim != 3 or visual_array.shape[0] != candidate.visual_channels:
        raise ValueError(
            f"visual contract mismatch: expected channels={candidate.visual_channels}, observed={visual_array.shape}"
        )
    expected_spectrum = (16, 128) if local else (128,)
    if tuple(spectrum_array.shape) != expected_spectrum:
        raise ValueError(
            f"spectrum contract mismatch: expected={expected_spectrum}, observed={tuple(spectrum_array.shape)}"
        )
    if not np.isfinite(visual_array).all() or not np.isfinite(spectrum_array).all():
        raise ValueError("cached input contains NaN or Infinity")

    visual = torch.from_numpy(visual_array.copy()).unsqueeze(0)
    spectrum = torch.from_numpy(spectrum_array.copy()).unsqueeze(0)
    try:
        source_ref = cache_file.relative_to(root).as_posix()
    except ValueError:
        source_ref = cache_file.name
    return InputFixture(
        visual=visual,
        spectrum=spectrum,
        source_ref=f"{dataset_key}:{source_ref}",
        representation_key=(
            f"{dataset_key}.rgb-local-spectrum16x128.legacy"
            if local
            else f"{dataset_key}.rgb-global-spectrum128.legacy"
        ),
        tensor_shapes={
            "rgb": tuple(int(x) for x in visual.shape[1:]),
            "spectrum": tuple(int(x) for x in spectrum.shape[1:]),
        },
        dtypes={"rgb": str(visual.dtype), "spectrum": str(spectrum.dtype)},
        channel_or_band_order={
            "rgb": ("legacy_visible_r", "legacy_visible_g", "legacy_visible_b"),
            "spectrum": ("legacy_resampled_band_order",),
        },
    )


def collect_companion_metadata(
    checkpoint: Path,
    runs_root: Path,
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Record checkpoint-bound and nearby preprocessing metadata."""

    references: list[str] = []
    parsed: list[tuple[str, Any]] = []
    current = checkpoint.parent
    while current == runs_root or runs_root in current.parents:
        for path in sorted(current.glob("*.json")):
            if path.stat().st_size > 16 * 1024 * 1024:
                continue
            ref = path.relative_to(runs_root.parent).as_posix()
            if ref in references:
                continue
            references.append(ref)
            try:
                parsed.append((ref, json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        if current == runs_root:
            break
        current = current.parent

    companion_fields: dict[str, Any] = {}
    checkpoint_fields: dict[str, Any] = {}
    preprocessing_tokens = ("preprocess", "normal", "cache", "mu", "sd", "pca", "transform", "fit_sample")

    def visit(value: Any, destination: dict[str, Any], prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if any(token in str(key).lower() for token in preprocessing_tokens):
                    if isinstance(child, (str, int, float, bool)) or child is None:
                        destination[path] = child
                visit(child, destination, path)
        elif isinstance(value, list):
            for index, child in enumerate(value[:100]):
                visit(child, destination, f"{prefix}[{index}]")

    visit(checkpoint_payload, checkpoint_fields)
    for _, value in parsed:
        visit(value, companion_fields)

    checkpoint_bound = bool(checkpoint_fields)
    status = (
        "checkpoint_bound"
        if checkpoint_bound
        else "companion_partial_unbound"
        if companion_fields
        else "missing"
    )
    return {
        "status": status,
        "checkpoint_bound": checkpoint_bound,
        "checkpoint_fields": checkpoint_fields,
        "companion_refs": references,
        "discovered_fields": companion_fields,
        "reason": (
            None
            if checkpoint_bound
            else "legacy checkpoint does not bind preprocessing config, fitted sample IDs, or transform hashes"
        ),
    }


def _source_hash(candidate: ModelCandidate, workspace: Path) -> str | None:
    path = workspace / candidate.source_ref
    return sha256_file(path) if path.is_file() else None


def audit_checkpoint(
    checkpoint: Path,
    *,
    workspace: Path,
    runs_root: Path,
    loader: ExternalCheckpointLoader,
    signature_index: Mapping[tuple[Any, ...], Sequence[ModelCandidate]],
    dataset_roots: Mapping[str, Path],
    device: Any,
) -> dict[str, Any]:
    """Audit one checkpoint without mutating it."""

    import torch

    checkpoint_ref = checkpoint.relative_to(workspace).as_posix()
    before_hash = sha256_file(checkpoint)
    record: dict[str, Any] = {
        "checkpoint_ref": checkpoint_ref,
        "size_bytes": checkpoint.stat().st_size,
        "source_reference": f"legacy-g-drive://{checkpoint_ref}",
        "source_sha256": before_hash,
        "restricted_deserialization": False,
        "checkpoint_unchanged": False,
    }
    try:
        payload = loader.load(
            checkpoint,
            source_reference=record["source_reference"],
            expected_content_hash=before_hash,
            map_location="cpu",
        )
        record["restricted_deserialization"] = True
        state = extract_model_state(payload)
        signature = state_signature(state)
        record["checkpoint_payload_keys"] = sorted(str(key) for key in payload.keys())
        record["state_dict"] = {
            "key_count": len(signature),
            "keys_sha256": hashlib.sha256(
                "\n".join(item[0] for item in signature).encode("utf-8")
            ).hexdigest(),
            "signature_sha256": hashlib.sha256(
                json.dumps(signature, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        candidate, alternatives = choose_candidate(
            signature_index.get(signature, ()), checkpoint_ref
        )
        record["model_identity_candidates"] = alternatives
        if candidate is None:
            raise LookupError("no known legacy model class has an exact state-dict signature")

        model = candidate.factory()
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"strict state load mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        record["model_class"] = f"{model.__class__.__module__}.{model.__class__.__qualname__}"
        record["model_source_ref"] = candidate.source_ref
        record["model_source_sha256"] = _source_hash(candidate, workspace)
        record["state_dict"]["strict_load"] = True

        dataset_key = infer_dataset_key(checkpoint_ref, candidate.output_dim)
        fixture = load_input_fixture(candidate, dataset_key, dataset_roots)
        record["input_contract"] = {
            "status": "validated",
            "dataset_key": dataset_key,
            "representation_key": fixture.representation_key,
            "modalities": ["rgb", "spectrum"],
            "tensor_shapes": fixture.tensor_shapes,
            "dtypes": fixture.dtypes,
            "channel_or_band_order": fixture.channel_or_band_order,
            "fixture_ref": fixture.source_ref,
        }
        record["preprocessing_metadata"] = collect_companion_metadata(
            checkpoint,
            runs_root,
            payload,
        )

        torch.cuda.reset_peak_memory_stats(device)
        model = model.to(device).eval()
        with torch.inference_mode():
            output = model(fixture.visual.to(device), fixture.spectrum.to(device))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"model forward returned {type(output).__name__}, not a tensor")
        torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(output).all().item())
        expected_output_shape = (1, candidate.output_dim)
        if tuple(output.shape) != expected_output_shape:
            raise ValueError(
                f"output contract mismatch: expected={expected_output_shape}, observed={tuple(output.shape)}"
            )
        if not finite:
            raise ValueError("CUDA forward output contains NaN or Infinity")
        record["cuda_forward"] = {
            "status": "pass",
            "device": torch.cuda.get_device_name(device),
            "output_shape": tuple(int(x) for x in output.shape),
            "output_dtype": str(output.dtype),
            "finite": finite,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        del model, output
        torch.cuda.empty_cache()

        if record["preprocessing_metadata"]["checkpoint_bound"]:
            record["compatibility_status"] = "compatible_for_inference"
            record["retraining_required"] = False
            record["retraining_reasons"] = []
        else:
            record["compatibility_status"] = "reloadable_but_metadata_incomplete"
            record["retraining_required"] = True
            record["retraining_reasons"] = [
                "preprocessing lineage is not bound into the legacy checkpoint",
                "formal v2 reuse cannot prove fitted-sample boundaries or transform identity",
            ]
    except (ExternalCheckpointLoadError, Exception) as exc:
        record["compatibility_status"] = "incompatible"
        record["retraining_required"] = True
        record["retraining_reasons"] = [
            "checkpoint failed controlled reload or bounded inference compatibility validation"
        ]
        record["failure"] = {
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }
        if isinstance(exc, ExternalCheckpointLoadError):
            record["failure"]["error_record"] = exc.error_record.model_dump(mode="json")
    finally:
        after_hash = sha256_file(checkpoint)
        record["checkpoint_sha256_after"] = after_hash
        record["checkpoint_unchanged"] = before_hash == after_hash
        if before_hash != after_hash:
            raise RuntimeError(f"legacy checkpoint changed during audit: {checkpoint_ref}")
    return record


def validate_roots(workspace: Path, runs_root: Path, dataset_roots: Mapping[str, Path], output_root: Path) -> None:
    """Enforce the G-drive-only read scope and workspace-relative output scope."""

    paths = [workspace, runs_root, output_root, *dataset_roots.values()]
    for path in paths:
        if path.drive.upper() != _ALLOWED_DRIVE:
            raise ValueError(f"non-G-drive path rejected: {path}")
    workspace_resolved = workspace.resolve()
    if runs_root.resolve() != (workspace_resolved / "runs"):
        raise ValueError("legacy runs root must be the workspace runs directory")
    if output_root.resolve() != (workspace_resolved / "runs_v2" / "audit"):
        raise ValueError("audit output must be workspace-relative runs_v2/audit")


def _atomic_json(path: Path, value: Any) -> None:
    """Publish a new JSON artifact atomically without overwriting a completed file."""

    if path.exists():
        raise FileExistsError(f"completed audit artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_checkpoint_audit(
    workspace: Path,
    *,
    runs_root: Path | None = None,
    output_root: Path | None = None,
    dataset_roots: Mapping[str, Path] | None = None,
) -> Path:
    """Run the complete bounded audit and return its report path."""

    import torch

    workspace = workspace.resolve()
    runs_root = (runs_root or workspace / "runs").resolve()
    output_root = (output_root or workspace / "runs_v2" / "audit").resolve()
    dataset_roots = dataset_roots or {
        "hyperleaf2024": Path(r"G:\Datasets\HyperLeaf2024"),
        "histology-hsi-gb": Path(r"G:\Datasets\PKG - HistologyHSI-GB"),
    }
    dataset_roots = {key: path.resolve() for key, path in dataset_roots.items()}
    validate_roots(workspace, runs_root, dataset_roots, output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for task 13.8")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5080" not in device_name:
        raise RuntimeError(f"expected RTX 5080, observed {device_name}")

    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("checkpoint-compatibility-%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    started = datetime.now(timezone.utc)
    candidates, unavailable_modules = build_candidates(workspace / "src")
    signature_index, candidate_failures = build_signature_index(candidates)
    checkpoints = sorted(
        path for path in runs_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES
    )
    loader = ExternalCheckpointLoader()
    records: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        record = audit_checkpoint(
            checkpoint,
            workspace=workspace,
            runs_root=runs_root,
            loader=loader,
            signature_index=signature_index,
            dataset_roots=dataset_roots,
            device=device,
        )
        records.append(record)
        if record["retraining_required"]:
            negative.append(
                {
                    "checkpoint_ref": record["checkpoint_ref"],
                    "compatibility_status": record["compatibility_status"],
                    "retraining_reasons": record["retraining_reasons"],
                    "failure": record.get("failure"),
                }
            )
        print(
            f"[{index}/{len(checkpoints)}] {record['compatibility_status']} {record['checkpoint_ref']}",
            flush=True,
        )

    completed = datetime.now(timezone.utc)
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["compatibility_status"])
        counts[status] = counts.get(status, 0) + 1
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "task_id": "13.8",
        "audit_id": run_id,
        "status": "pass" if records and counts.get("incompatible", 0) == 0 else "completed_with_negative_evidence",
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_utc": completed.isoformat().replace("+00:00", "Z"),
        "duration_ms": int((completed - started).total_seconds() * 1000),
        "scope": {
            "workspace": "G:/Datasets/HyperCross",
            "read_root": "runs",
            "output_ref": f"runs_v2/audit/{run_id}",
            "f_drive_references_followed": False,
            "legacy_files_opened_for_write": False,
        },
        "environment": {
            "python_executable": sys.executable,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": device_name,
        },
        "checks": {
            "all_checkpoints_hash_gated_before_deserialization": all(record["restricted_deserialization"] for record in records),
            "all_checkpoint_hashes_unchanged": all(record["checkpoint_unchanged"] for record in records),
            "all_paths_on_g_drive": True,
            "output_workspace_relative": True,
            "single_batch_cuda_forward_bounded": True,
        },
        "summary": {
            "checkpoint_count": len(records),
            "status_counts": counts,
            "negative_evidence_count": len(negative),
            "known_candidate_count": len(candidates),
            "state_signature_count": len(signature_index),
        },
        "candidate_import_failures": unavailable_modules,
        "candidate_construction_failures": candidate_failures,
        "requirements": ["1.1-1.5", "6.3-6.13", "13.33-13.36", "17.48-17.53"],
        "checkpoints": records,
    }
    report_digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    report["report_content_sha256"] = report_digest
    _atomic_json(run_dir / "checkpoint_compatibility_audit.json", report)

    negative_path = run_dir / "negative_evidence.jsonl"
    if negative_path.exists():
        raise FileExistsError(negative_path)
    temporary = negative_path.with_name(f".{negative_path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        for item in negative:
            stream.write(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(negative_path)
    return run_dir / "checkpoint_compatibility_audit.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(r"G:\Datasets\HyperCross"))
    args = parser.parse_args(argv)
    report = run_checkpoint_audit(args.workspace)
    print(f"AUDIT_REPORT={report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
