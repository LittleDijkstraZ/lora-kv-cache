"""Canonical record types for the generative-QA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GENERATIVE_TASK_FAMILY = "generative_task"

_TASK_FAMILY_ALIASES = {
    GENERATIVE_TASK_FAMILY: GENERATIVE_TASK_FAMILY,
    "generative_qa": GENERATIVE_TASK_FAMILY,
}

_TASK_FAMILIES = set(_TASK_FAMILY_ALIASES.values())


def normalize_task_family(task_family: str) -> str:
    """Normalize supported generative-QA task-family labels."""

    normalized = _TASK_FAMILY_ALIASES.get(str(task_family))
    if normalized is None:
        raise ValueError(
            f"Unknown task_family={task_family!r}; "
            f"expected one of {sorted(_TASK_FAMILIES)}"
        )
    return normalized


@dataclass
class CanonicalQARecord:
    """Shared fields carried by normalized generative-QA records."""

    record_id: str
    task_family: str
    question: str
    context: str
    cluster_idx: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.record_id = str(self.record_id)
        self.question = str(self.question)
        self.context = str(self.context)
        self.task_family = normalize_task_family(self.task_family)
        if self.cluster_idx is not None:
            self.cluster_idx = int(self.cluster_idx)
        self.metadata = dict(self.metadata or {})


@dataclass
class CanonicalGenerativeQARecord(CanonicalQARecord):
    """Canonical form for a generative-QA example."""

    answers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        answers = [str(a) for a in (self.answers or []) if a is not None]
        if not answers:
            raise ValueError("CanonicalGenerativeQARecord requires at least one answer")
        self.answers = answers
