"""Environment capture and immutable environment-lock records."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
from typing import Any, Callable, Optional

from pydantic import ConfigDict, Field, model_validator

from .base import ContentHash, StrictSchemaModel
from .canonical import hash_object


class FrozenSchemaModel(StrictSchemaModel):
    """Strict schema record whose fields cannot change after validation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class EnvironmentCaptureError(RuntimeError):
    """Raised when required dependency versions cannot be captured."""

    def __init__(self, missing_distributions: list[str]) -> None:
        self.missing_distributions = tuple(sorted(missing_distributions))
        joined = ", ".join(self.missing_distributions)
        super().__init__(f"required distributions are not installed: {joined}")


class EnvironmentLock(FrozenSchemaModel):
    """A hash-bound snapshot of benchmark software and visible compute devices."""

    python_version: str = Field(min_length=1)
    pytorch_version: str = Field(min_length=1)
    numpy_version: str = Field(min_length=1)
    pandas_version: str = Field(min_length=1)
    pillow_version: str = Field(min_length=1)
    tifffile_version: str = Field(min_length=1)
    sklearn_version: str = Field(min_length=1)
    scipy_version: str = Field(min_length=1)
    cuda_version: Optional[str] = None
    cuda_available: bool
    cuda_device_count: int = Field(strict=True, ge=0)
    device_identity: str = Field(min_length=1)
    lock_hash: ContentHash

    @classmethod
    def create(cls, **values: Any) -> "EnvironmentLock":
        """Create a lock and bind its hash to every captured field."""

        payload = dict(values)
        payload["lock_hash"] = hash_object(payload)
        return cls.model_validate(payload)

    def hash_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"lock_hash"})

    @model_validator(mode="after")
    def validate_lock(self) -> "EnvironmentLock":
        if self.lock_hash != hash_object(self.hash_payload()):
            raise ValueError("lock_hash does not match the captured environment")
        if self.cuda_available != (self.cuda_device_count > 0):
            raise ValueError("cuda_available must agree with cuda_device_count")
        return self


_DISTRIBUTION_FIELDS = {
    "pytorch_version": "torch",
    "numpy_version": "numpy",
    "pandas_version": "pandas",
    "pillow_version": "Pillow",
    "tifffile_version": "tifffile",
    "sklearn_version": "scikit-learn",
    "scipy_version": "scipy",
}


def capture_environment_lock(
    *,
    version_provider: Callable[[str], str] = importlib.metadata.version,
    torch_module: Any = None,
) -> EnvironmentLock:
    """Capture dependency, CUDA build, and visible device identity.

    PyTorch is imported lazily so importing schema modules does not initialize
    CUDA. Missing required distributions are returned together in one error.
    """

    versions: dict[str, str] = {}
    missing: list[str] = []
    for field_name, distribution_name in _DISTRIBUTION_FIELDS.items():
        try:
            versions[field_name] = version_provider(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution_name)
    if missing:
        raise EnvironmentCaptureError(missing)

    torch = torch_module if torch_module is not None else importlib.import_module("torch")
    cuda_build_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    if cuda_available:
        identities = [
            f"cuda:{index}:{torch.cuda.get_device_name(index)}"
            for index in range(device_count)
        ]
        device_identity = ";".join(identities)
    else:
        device_identity = f"cpu:{platform.machine() or 'unknown'}"

    return EnvironmentLock.create(
        python_version=platform.python_version(),
        **versions,
        cuda_version=str(cuda_build_version) if cuda_build_version is not None else None,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        device_identity=device_identity,
    )
