"""Append-only candidate-space registration with a pre-evaluation seal."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import datetime

from benchmark.schema.candidate_space import CandidateSpace, FrozenCandidateSpaceRegistry
from benchmark.schema.canonical import normalize_id


class CandidateSpaceRegistryError(RuntimeError):
    """Raised when candidate-space registration or freezing is invalid."""


class CandidateSpaceRegistry(Mapping[str, CandidateSpace]):
    """Collect complete candidate spaces and seal them before evaluation.

    Registration is append-only. ``freeze_before_evaluation`` permanently closes
    this mutable builder and returns the only artifact intended for evaluation
    consumers.
    """

    def __init__(self) -> None:
        self._spaces: dict[str, CandidateSpace] = {}
        self._frozen: FrozenCandidateSpaceRegistry | None = None

    def __getitem__(self, candidate_space_key: str) -> CandidateSpace:
        return self.resolve(candidate_space_key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.candidate_space_keys)

    def __len__(self) -> int:
        return len(self._spaces)

    @property
    def candidate_space_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._spaces, key=lambda value: value.encode("utf-8")))

    @property
    def is_frozen_for_evaluation(self) -> bool:
        return self._frozen is not None

    def register(self, candidate_space: CandidateSpace) -> None:
        """Append one already hash-bound candidate space without replacement."""

        if self._frozen is not None:
            raise CandidateSpaceRegistryError(
                "candidate-space registry is frozen; evaluation candidates cannot change"
            )
        if not isinstance(candidate_space, CandidateSpace):
            raise TypeError("candidate_space must be a CandidateSpace")
        candidate_space.require_integrity()
        key = candidate_space.candidate_space_key
        if key in self._spaces:
            raise CandidateSpaceRegistryError(
                f"candidate space {key!r} is already registered"
            )
        self._spaces[key] = candidate_space

    def resolve(self, candidate_space_key: str) -> CandidateSpace:
        key = normalize_id(candidate_space_key)
        try:
            space = self._spaces[key]
        except KeyError as exc:
            raise KeyError(f"candidate space {key!r} is not registered") from exc
        space.require_integrity()
        return space

    def freeze_before_evaluation(
        self,
        *,
        frozen_utc: datetime | None = None,
    ) -> FrozenCandidateSpaceRegistry:
        """Seal all registrations and return their immutable hash-bound snapshot."""

        if self._frozen is not None:
            return self._frozen
        if not self._spaces:
            raise CandidateSpaceRegistryError(
                "at least one candidate space must be registered before evaluation"
            )
        self._frozen = FrozenCandidateSpaceRegistry.create(
            self._spaces.values(), frozen_utc=frozen_utc
        )
        return self._frozen


__all__ = ["CandidateSpaceRegistry", "CandidateSpaceRegistryError"]
