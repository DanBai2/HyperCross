"""Dataset adapter interfaces for the three independent benchmark tracks.

Adapters receive their storage configuration in concrete implementations. These
protocols intentionally contain no paths, environment assumptions, or data
materialization behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from benchmark.schema.datasets import DatasetSpec, SampleRecord


@runtime_checkable
class DatasetAdapter(Protocol):
    """Common interface for a configured, streaming dataset adapter."""

    @property
    def spec(self) -> DatasetSpec:
        """Return the registered semantics for this configured source."""

    def iter_samples(self) -> Iterable[SampleRecord]:
        """Yield sample metadata without requiring full dataset materialization."""


@runtime_checkable
class HyperLeafDatasetAdapter(DatasetAdapter, Protocol):
    """Adapter contract exposing official HyperLeaf train/test partitions."""

    def iter_training_samples(self) -> Iterable[SampleRecord]:
        """Yield the labeled internal-selection and refit samples."""

    def iter_official_test_samples(self) -> Iterable[SampleRecord]:
        """Yield official-test metadata without granting label access."""


@runtime_checkable
class HistologyDatasetAdapter(DatasetAdapter, Protocol):
    """Adapter contract for patient-grouped Histology sample metadata."""

    def iter_patient_grouped_samples(self) -> Iterable[SampleRecord]:
        """Yield ROI samples carrying externally configured patient groups."""


@runtime_checkable
class RemoteSanityDatasetAdapter(DatasetAdapter, Protocol):
    """Adapter contract for the isolated remote-sensing sanity track."""

    def iter_remote_scene_samples(self) -> Iterable[SampleRecord]:
        """Yield scene samples for the separately governed sanity track."""
