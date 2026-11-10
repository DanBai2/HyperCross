"""Append-only run, aggregate, artifact, and error journal interfaces."""

from benchmark.registry.candidate_space import (
    CandidateSpaceRegistry,
    CandidateSpaceRegistryError,
)
from benchmark.registry.error_journal import ErrorJournal, ErrorJournalWriteError
from benchmark.registry.records import (
    AggregateRecord,
    ArtifactRecord,
    ClaimCeiling,
    ClaimDecision,
    FrozenRunIdentity,
    InputContract,
    LegacySourceReference,
    ModelIdentity,
    NegativeEvidenceReason,
    NegativeEvidenceRecord,
    Provenance,
    RecoveryArtifact,
    RunRecord,
    SupersedingRecord,
)
from benchmark.registry.run_registry import (
    AppendOnlyRunRegistry,
    RegistryCorruptionError,
    RegistryError,
    RegistryInvariantError,
    RegistryWriteError,
    RunRegistry,
)

__all__ = [
    "AggregateRecord",
    "AppendOnlyRunRegistry",
    "ArtifactRecord",
    "CandidateSpaceRegistry",
    "CandidateSpaceRegistryError",
    "ClaimCeiling",
    "ClaimDecision",
    "ErrorJournal",
    "ErrorJournalWriteError",
    "FrozenRunIdentity",
    "InputContract",
    "LegacySourceReference",
    "ModelIdentity",
    "NegativeEvidenceReason",
    "NegativeEvidenceRecord",
    "Provenance",
    "RecoveryArtifact",
    "RegistryCorruptionError",
    "RegistryError",
    "RegistryInvariantError",
    "RegistryWriteError",
    "RunRecord",
    "RunRegistry",
    "SupersedingRecord",
]
