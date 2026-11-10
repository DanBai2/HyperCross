"""Unit tests for canonical serialization and content hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum

import pytest

from benchmark.schema import (
    CanonicalCodec,
    CanonicalEncodingError,
    NonFiniteFloatError,
    PublicationValueError,
    canonical_serialize,
    encode_binary64,
    encode_utc_timestamp,
    hash_file,
    hash_object,
    normalize_id,
)


class ExampleKind(Enum):
    FIRST = "first"


@dataclass(frozen=True)
class ExampleRecord:
    name: str
    score: float


def test_mapping_order_and_unicode_forms_have_identical_bytes_and_hashes() -> None:
    first = {"z": 1, "e\u0301": "Cafe\u0301"}
    second = {"é": "Café", "z": 1}

    assert canonical_serialize(first) == canonical_serialize(second)
    assert hash_object(first) == hash_object(second)
    assert canonical_serialize(first).decode("utf-8").find("é") >= 0


def test_sets_are_sorted_by_canonical_element_bytes() -> None:
    assert canonical_serialize({"z", "a", "é"}) == canonical_serialize(
        frozenset({"é", "z", "a"})
    )


def test_sequences_preserve_semantic_order() -> None:
    assert canonical_serialize([1, 2]) != canonical_serialize([2, 1])
    assert canonical_serialize([1, 2]) == canonical_serialize((1, 2))


def test_ids_normalize_unicode_and_path_separators_without_case_folding() -> None:
    assert normalize_id("Group\\Cafe\u0301\\A") == "Group/Café/A"
    assert normalize_id("Sample-A") != normalize_id("sample-a")
    assert CanonicalCodec.normalize_id("x\\y") == "x/y"


def test_binary64_encoding_is_typed_exact_at_finite_boundaries() -> None:
    assert encode_binary64(1.0) == "0x3ff0000000000000"
    assert encode_binary64(0.0) == "0x0000000000000000"
    assert encode_binary64(-0.0) == "0x8000000000000000"
    assert encode_binary64(float.fromhex("0x1.0000000000000p-1022")) == (
        "0x0010000000000000"
    )
    assert encode_binary64(float.fromhex("0x0.0000000000001p-1022")) == (
        "0x0000000000000001"
    )
    assert encode_binary64(float.fromhex("0x1.fffffffffffffp+1023")) == (
        "0x7fefffffffffffff"
    )
    assert canonical_serialize(1) != canonical_serialize(1.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_binary64_encoding_rejects_every_non_finite_boundary(value: float) -> None:
    with pytest.raises(NonFiniteFloatError):
        canonical_serialize(value)


def test_timestamp_is_converted_to_utc_with_fixed_microseconds() -> None:
    source = datetime(
        2027,
        1,
        2,
        3,
        4,
        5,
        6,
        tzinfo=timezone(timedelta(hours=8)),
    )
    assert encode_utc_timestamp(source) == "2027-01-01T19:04:05.000006Z"

    with pytest.raises(CanonicalEncodingError, match="UTC offset"):
        encode_utc_timestamp(datetime(2027, 1, 1))


def test_enum_and_typed_records_are_stably_encoded() -> None:
    value = {
        "kind": ExampleKind.FIRST,
        "record": ExampleRecord(name="Cafe\u0301", score=0.5),
    }
    encoded = canonical_serialize(value)

    assert encoded == CanonicalCodec.serialize(value)
    assert hash_object(value) == CanonicalCodec.hash(value)
    assert b"enum" in encoded
    assert "Café" in encoded.decode("utf-8")


def test_publication_decimal_is_rejected_from_scientific_hashing() -> None:
    with pytest.raises(PublicationValueError, match="raw binary64 metric"):
        hash_object({"metric": Decimal("0.5000")})


def test_file_hash_streams_raw_bytes_with_configurable_chunks(tmp_path) -> None:
    content = ("Café/" * 1000).encode("utf-8")
    source = tmp_path / "artifact with spaces.bin"
    source.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert hash_file(source, chunk_size=17) == expected
    assert hash_file(str(source)) == expected

    with pytest.raises(ValueError, match="positive integer"):
        hash_file(source, chunk_size=0)


def test_ambiguous_or_cyclic_values_are_rejected() -> None:
    with pytest.raises(CanonicalEncodingError, match="keys must be strings"):
        canonical_serialize({1: "value"})

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(CanonicalEncodingError, match="cyclic"):
        canonical_serialize(cyclic)
