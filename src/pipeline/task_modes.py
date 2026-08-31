"""Task-mode helpers shared by synthesis, training, and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


QA_SHORT_TASK = "qa_short"
QA_TASK_MODES = frozenset({QA_SHORT_TASK})
ALL_TASK_MODES = QA_TASK_MODES
TASK_FAMILY_QA = "qa"

_BOXED_SHORT_INSTRUCTION = (
    '\nAnswer the question (one word or a few words) by putting the answer '
    'inside the brackets of the "\\boxed{}" notation.'
)


@dataclass(frozen=True)
class TaskModeSpec:
    task_mode: str
    task_family: str
    qa_instruction: str
    synthesis_answer_guidance: str


TASK_MODE_SPECS = {
    QA_SHORT_TASK: TaskModeSpec(
        task_mode=QA_SHORT_TASK,
        task_family=TASK_FAMILY_QA,
        qa_instruction=_BOXED_SHORT_INSTRUCTION,
        synthesis_answer_guidance=(
            "Each answer must be a short factual phrase grounded in the passage. "
            "Prefer the passage's exact wording when possible. "
        ),
    )
}


def normalize_task_mode(task_mode: str) -> str:
    normalized = str(task_mode).strip().lower()
    if normalized == "qa":
        return QA_SHORT_TASK
    if normalized not in ALL_TASK_MODES:
        raise ValueError(f"Unknown task mode={task_mode!r}; expected 'qa_short'.")
    return normalized


def normalize_task_modes(task: Any, *, default: list[str] | None = None) -> list[str]:
    if task is None:
        raw_values: list[Any] = list(default or [])
    elif isinstance(task, str):
        raw_values = [task]
    elif isinstance(task, Iterable) and not isinstance(task, dict):
        raw_values = list(task)
    else:
        raw_values = [task]
    result: list[str] = []
    for value in raw_values:
        if value is None:
            continue
        canonical = normalize_task_mode(str(value))
        if canonical not in result:
            result.append(canonical)
    return result or list(default or [])


def task_mode_family(task_mode: str) -> str:
    return TASK_MODE_SPECS[normalize_task_mode(task_mode)].task_family


def is_qa_task_mode(task_mode: str) -> bool:
    return normalize_task_mode(task_mode) in QA_TASK_MODES


def validate_eval_task_mode(task_mode: str) -> str:
    return normalize_task_mode(task_mode)


def get_task_mode_spec(task_mode: str) -> TaskModeSpec:
    return TASK_MODE_SPECS[normalize_task_mode(task_mode)]


def get_qa_boxed_instruction(task_mode: str) -> str:
    return get_task_mode_spec(task_mode).qa_instruction


def cache_task_label(task_modes: list[str]) -> str:
    normalize_task_modes(task_modes)
    return "qa"
