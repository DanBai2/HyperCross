from pathlib import Path

import pytest
import torch

from benchmark.audit.checkpoint_compatibility import (
    ModelCandidate,
    choose_candidate,
    extract_model_state,
    sha256_file,
    state_signature,
    validate_roots,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def _candidate(name: str) -> ModelCandidate:
    return ModelCandidate(
        name=name,
        factory=TinyModel,
        output_dim=1,
        spectrum_kind="global_128",
        visual_channels=3,
        source_ref="src/tiny.py",
    )


def test_state_signature_captures_exact_keys_shapes_and_dtypes() -> None:
    state = TinyModel().state_dict()

    signature = state_signature(state)

    assert signature == (
        ("linear.weight", (1, 2), "torch.float32"),
        ("linear.bias", (1,), "torch.float32"),
    )
    assert extract_model_state({"model": state, "epoch": 1}) is state


def test_extract_model_state_rejects_non_tensor_entries() -> None:
    with pytest.raises(TypeError, match="not tensor-like"):
        extract_model_state({"model": {"linear.weight": "unsafe"}})


def test_choose_candidate_uses_checkpoint_name_hint_deterministically() -> None:
    route = _candidate("hypercross_routeA.CrossModalRouteA[out_dim=2]")
    token = _candidate("hypercross_routeA.TokenRouteA[out_dim=2]")

    selected, alternatives = choose_candidate(
        [token, route],
        "runs/hist_example/routeA_best.pt",
    )

    assert selected is route
    assert alternatives[0] == route.name


def test_sha256_file_is_stable_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"legacy-checkpoint")
    before = path.read_bytes()

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert path.read_bytes() == before


def test_validate_roots_rejects_any_f_drive_reference() -> None:
    with pytest.raises(ValueError, match="non-G-drive path rejected"):
        validate_roots(
            Path(r"G:\Datasets\HyperCross"),
            Path(r"G:\Datasets\HyperCross\runs"),
            {
                "hyperleaf2024": Path(r"G:\Datasets\HyperLeaf2024"),
                "histology-hsi-gb": Path(r"F:\forbidden"),
            },
            Path(r"G:\Datasets\HyperCross\runs_v2\audit"),
        )
