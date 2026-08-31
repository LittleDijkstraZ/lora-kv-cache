"""Stable document identity helpers for generative QA."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def _flatten_context_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        titles = value.get("title")
        sentences = value.get("sentences")
        if isinstance(titles, Sequence) and isinstance(sentences, Sequence):
            parts: list[str] = []
            for title, sents in zip(titles, sentences):
                if title:
                    parts.append(str(title))
                if isinstance(sents, Sequence) and not isinstance(sents, str):
                    parts.append(" ".join(str(sent) for sent in sents))
                elif sents:
                    parts.append(str(sents))
            if parts:
                return "\n".join(parts)
        return json.dumps(dict(value), sort_keys=True, ensure_ascii=False)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(_flatten_context_value(item) for item in value)
    return str(value)


def normalize_document_text(value: Any) -> str:
    """Collapse context-ish inputs into a stable whitespace-normalized string."""

    flattened = _flatten_context_value(value)
    return _WHITESPACE_RE.sub(" ", flattened).strip()


def hash_document_text(value: Any) -> str:
    normalized = normalize_document_text(value)
    if not normalized:
        raise ValueError("Cannot hash an empty document text.")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"ctxsha1_{digest[:16]}"


def resolve_document_id(record: Mapping[str, Any]) -> str:
    """Resolve a stable document id without changing existing record contracts.

    Explicit document and patient ids take precedence; otherwise the normalized
    document text is hashed.
    """

    metadata = record.get("metadata") or {}
    if isinstance(metadata, Mapping):
        for key in ("document_id", "doc_id", "patient_id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

    for key in ("document_id", "doc_id", "patient_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    for key in ("full_context", "context", "chunk_text"):
        value = record.get(key)
        if normalize_document_text(value):
            return hash_document_text(value)

    raise ValueError("Could not resolve document id from record.")


__all__ = [
    "hash_document_text",
    "normalize_document_text",
    "resolve_document_id",
]
