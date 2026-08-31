"""Exact-match and ROUGE-L metrics for generative QA."""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match_score(prediction: str, gold_answers: list[str]) -> float:
    prediction = normalize_answer(prediction)
    return float(any(prediction == normalize_answer(answer) for answer in gold_answers))


def extract_boxed_answer(text: str) -> str:
    match = re.search(r"\\boxed\{([^}]*)\}", str(text))
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else str(text).strip()


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_score(prediction: str, gold_answers: list[str]) -> float:
    prediction_tokens = _tokens(prediction)
    if not prediction_tokens:
        return 0.0
    best = 0.0
    for answer in gold_answers:
        answer_tokens = _tokens(answer)
        if not answer_tokens:
            continue
        overlap = _lcs_length(prediction_tokens, answer_tokens)
        if overlap:
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(answer_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


__all__ = [
    "exact_match_score",
    "extract_boxed_answer",
    "normalize_answer",
    "rouge_l_score",
]
