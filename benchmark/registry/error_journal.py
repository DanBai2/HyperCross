"""Durable append-only journal for machine-readable benchmark failures."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Union

from benchmark.schema.records import ErrorRecord


class ErrorJournalWriteError(OSError):
    """Raised when an error record cannot be durably appended."""


class ErrorJournal:
    """Append :class:`ErrorRecord` values as one durable JSON object per line.

    The journal never opens its file with truncation semantics. A process-local
    lock keeps records from interleaving between threads, while ``O_APPEND``
    preserves append behavior at the operating-system boundary.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the journal path without opening it."""

        return self._path

    def append(self, error: ErrorRecord) -> None:
        """Append and flush one record without rewriting prior records."""

        if not isinstance(error, ErrorRecord):
            raise TypeError("error must be an ErrorRecord")

        payload = json.dumps(
            error.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"

        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    self._path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("journal append made no progress")
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise ErrorJournalWriteError(
                    f"could not append error journal {self._path}"
                ) from exc
