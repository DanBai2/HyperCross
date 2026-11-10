"""Canonical serialization and SHA-256 content hashing.

The codec represents schema-bound Python values as compact, typed JSON.  Type
markers keep values such as ``1``, ``1.0``, and ``"1"`` distinct, while the
JSON envelope provides a portable UTF-8 byte representation.

Scientific hashes must be computed from raw schema values.  Publication-only
fixed-decimal projections (for example ``Decimal("0.5000")``) are deliberately
unsupported so display rounding cannot silently replace binary64 metric data
in an artifact identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from collections.abc import Mapping, Set as AbstractSet
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Set, Union

_SHA256_HEX_LENGTH = 64
_DEFAULT_CHUNK_SIZE = 1024 * 1024


class CanonicalEncodingError(ValueError):
    """Raised when a value cannot be represented canonically."""


class NonFiniteFloatError(CanonicalEncodingError):
    """Raised when NaN or infinity is supplied to the codec."""


class PublicationValueError(CanonicalEncodingError):
    """Raised when a publication-formatted value enters scientific hashing."""


def normalize_id(value: str) -> str:
    """Normalize an identifier without changing its case.

    Identifier normalization applies Unicode NFC and converts Windows path
    separators to ``/``.  It intentionally performs no case folding, trimming,
    or path resolution because each of those operations could merge distinct
    dataset identifiers.
    """

    if not isinstance(value, str):
        raise TypeError("identifier must be a string")
    return unicodedata.normalize("NFC", value).replace("\\", "/")


def encode_utc_timestamp(value: datetime) -> str:
    """Encode an aware datetime as UTC RFC 3339 with six fractional digits."""

    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise CanonicalEncodingError("timestamp must include a UTC offset")
    utc_value = value.astimezone(timezone.utc)
    return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def encode_binary64(value: float) -> str:
    """Return the exact big-endian IEEE-754 binary64 bit pattern."""

    if not isinstance(value, float):
        raise TypeError("binary64 value must be a float")
    if not math.isfinite(value):
        raise NonFiniteFloatError("NaN and infinity are not canonical values")
    return "0x" + struct.pack(">d", value).hex()


def _enum_token(value: Enum) -> str:
    token = value.value if isinstance(value.value, str) else value.name
    if not isinstance(token, str) or not token:
        raise CanonicalEncodingError("enum token must be a non-empty string")
    return unicodedata.normalize("NFC", token)


def _model_fields(value: Any) -> Union[Mapping[str, Any], None]:
    """Extract declared fields from supported typed model instances."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", round_trip=True)
        if not isinstance(dumped, Mapping):
            raise CanonicalEncodingError("typed model dump must be a mapping")
        return dumped

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}

    return None


def _canonical_mapping(value: Mapping[str, Any], active: Set[int]) -> List[Any]:
    normalized: Dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CanonicalEncodingError("canonical map keys must be strings")
        normalized_key = unicodedata.normalize("NFC", key)
        if normalized_key in normalized:
            raise CanonicalEncodingError(
                f"map keys collide after Unicode normalization: {normalized_key!r}"
            )
        normalized[normalized_key] = item

    ordered_keys = sorted(normalized, key=lambda key: key.encode("utf-8"))
    return [
        "map",
        [[key, _canonical_node(normalized[key], active)] for key in ordered_keys],
    ]


def _canonical_set(value: AbstractSet[Any], active: Set[int]) -> List[Any]:
    encoded_items = []
    seen = set()
    for item in value:
        node = _canonical_node(item, active)
        encoded = _dump_node(node)
        if encoded in seen:
            raise CanonicalEncodingError(
                "set elements collide after canonical normalization"
            )
        seen.add(encoded)
        encoded_items.append((encoded, node))
    encoded_items.sort(key=lambda pair: pair[0])
    return ["set", [node for _, node in encoded_items]]


def _canonical_node(value: Any, active: Set[int]) -> List[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, Enum):
        return ["enum", _enum_token(value)]
    if isinstance(value, bool):
        return ["boolean", value]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, float):
        return ["binary64", encode_binary64(value)]
    if isinstance(value, str):
        return ["string", unicodedata.normalize("NFC", value)]
    if isinstance(value, datetime):
        return ["timestamp", encode_utc_timestamp(value)]
    if isinstance(value, Decimal):
        raise PublicationValueError(
            "Decimal publication projections are excluded from scientific hashes; "
            "hash the raw binary64 metric instead"
        )

    model_fields = _model_fields(value)
    if model_fields is not None:
        return _with_cycle_guard(value, active, _canonical_mapping, model_fields)
    if isinstance(value, Mapping):
        return _with_cycle_guard(value, active, _canonical_mapping, value)
    if isinstance(value, (list, tuple)):
        return _with_cycle_guard(
            value,
            active,
            lambda sequence, guarded: [
                "sequence", [_canonical_node(item, guarded) for item in sequence]
            ],
            value,
        )
    if isinstance(value, (set, frozenset)):
        return _with_cycle_guard(value, active, _canonical_set, value)

    raise CanonicalEncodingError(
        f"unsupported canonical value type: {type(value).__module__}.{type(value).__qualname__}"
    )


def _with_cycle_guard(
    owner: Any,
    active: Set[int],
    encoder: Any,
    value: Any,
) -> List[Any]:
    owner_id = id(owner)
    if owner_id in active:
        raise CanonicalEncodingError("cyclic values are not canonical")
    active.add(owner_id)
    try:
        return encoder(value, active)
    finally:
        active.remove(owner_id)


def _dump_node(node: List[Any]) -> bytes:
    text = json.dumps(
        node,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def canonical_serialize(value: Any) -> bytes:
    """Serialize a schema-bound value to deterministic UTF-8 bytes."""

    return _dump_node(_canonical_node(value, set()))


def _decode_canonical_node(node: Any) -> Any:
    """Decode one typed canonical node while rejecting malformed envelopes."""

    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise CanonicalEncodingError("canonical node must be a tagged array")
    tag = node[0]
    if tag == "null" and node == ["null"]:
        return None
    if tag == "boolean" and len(node) == 2 and isinstance(node[1], bool):
        return node[1]
    if tag == "integer" and len(node) == 2 and isinstance(node[1], str):
        if re.fullmatch(r"0|-?[1-9][0-9]*", node[1]) is None:
            raise CanonicalEncodingError("canonical integer has an invalid encoding")
        return int(node[1])
    if tag == "binary64" and len(node) == 2 and isinstance(node[1], str):
        if re.fullmatch(r"0x[0-9a-f]{16}", node[1]) is None:
            raise CanonicalEncodingError("canonical binary64 has an invalid encoding")
        value = struct.unpack(">d", bytes.fromhex(node[1][2:]))[0]
        if not math.isfinite(value):
            raise NonFiniteFloatError("NaN and infinity are not canonical values")
        return value
    if tag in {"string", "enum"} and len(node) == 2 and isinstance(node[1], str):
        value = unicodedata.normalize("NFC", node[1])
        if tag == "enum" and not value:
            raise CanonicalEncodingError("canonical enum token must not be empty")
        return value
    if tag == "timestamp" and len(node) == 2 and isinstance(node[1], str):
        try:
            value = datetime.fromisoformat(node[1].replace("Z", "+00:00"))
        except ValueError as exc:
            raise CanonicalEncodingError("canonical timestamp is invalid") from exc
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalEncodingError("canonical timestamp must include a UTC offset")
        return value
    if tag == "sequence" and len(node) == 2 and isinstance(node[1], list):
        return tuple(_decode_canonical_node(item) for item in node[1])
    if tag == "set" and len(node) == 2 and isinstance(node[1], list):
        decoded = tuple(_decode_canonical_node(item) for item in node[1])
        try:
            result = frozenset(decoded)
        except TypeError as exc:
            raise CanonicalEncodingError("canonical set contains an unhashable value") from exc
        if len(result) != len(decoded):
            raise CanonicalEncodingError("canonical set contains duplicate values")
        return result
    if tag == "map" and len(node) == 2 and isinstance(node[1], list):
        result: Dict[str, Any] = {}
        for entry in node[1]:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
            ):
                raise CanonicalEncodingError("canonical map entry must be [string, value]")
            key = unicodedata.normalize("NFC", entry[0])
            if key in result:
                raise CanonicalEncodingError(
                    f"canonical map contains duplicate key: {key!r}"
                )
            result[key] = _decode_canonical_node(entry[1])
        return result
    raise CanonicalEncodingError(f"unsupported or malformed canonical tag: {tag!r}")


def canonical_deserialize(content: Union[bytes, bytearray, str]) -> Any:
    """Decode typed canonical JSON into Python values.

    Schema-specific reload APIs remain responsible for validating the decoded
    values and for checking that reserialization exactly matches the input.
    """

    if isinstance(content, bytearray):
        content = bytes(content)
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalEncodingError("canonical content must be UTF-8") from exc
    elif isinstance(content, str):
        text = content
    else:
        raise TypeError("canonical content must be bytes, bytearray, or str")
    try:
        node = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CanonicalEncodingError("canonical content is not valid JSON") from exc
    return _decode_canonical_node(node)


def hash_object(value: Any) -> str:
    """Return the lowercase SHA-256 hash of a canonical object serialization."""

    return hashlib.sha256(canonical_serialize(value)).hexdigest()


def _hash_stream(stream: BinaryIO, chunk_size: int) -> str:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    result = digest.hexdigest()
    if len(result) != _SHA256_HEX_LENGTH:  # defensive invariant
        raise RuntimeError("unexpected SHA-256 digest length")
    return result


def hash_file(path: Union[str, Path], *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """Stream a file and return its lowercase SHA-256 content hash."""

    with Path(path).open("rb") as stream:
        return _hash_stream(stream, chunk_size)


class CanonicalCodec:
    """Namespace API matching the benchmark design interface."""

    @staticmethod
    def serialize(value: Any) -> bytes:
        return canonical_serialize(value)

    @staticmethod
    def deserialize(content: Union[bytes, bytearray, str]) -> Any:
        return canonical_deserialize(content)

    @staticmethod
    def hash(value: Any) -> str:
        return hash_object(value)

    @staticmethod
    def normalize_id(value: str) -> str:
        return normalize_id(value)
