#!/usr/bin/env python3
"""Prompting and output parsing helpers for synthetic QA generation."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from src.pipeline.task_modes import (
    QA_SHORT_TASK,
    get_task_mode_spec,
)

log = logging.getLogger(__name__)

_DEFAULT_LLAMA3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


def ensure_chat_template(tokenizer) -> None:
    """Assign a default Llama-3 style chat template if the tokenizer has none."""
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = _DEFAULT_LLAMA3_CHAT_TEMPLATE
        log.warning(
            "Tokenizer has no chat_template - applied default Llama-3 style template."
        )


def _supports_kwarg(tokenizer, kwarg_name: str) -> bool:
    """Return True if tokenizer.apply_chat_template accepts kwarg_name."""
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return False
    if kwarg_name in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def build_prompt(
    tokenizer,
    chunk_text: str,
    pairs_per_call: int,
    surrounding_context: str | None = None,
) -> str:
    """Build the generation prompt for one chunk."""
    ensure_chat_template(tokenizer)

    if pairs_per_call == 1:
        pair_instruction = "1 item"
    else:
        pair_instruction = f"{pairs_per_call} items"

    answer_guidance = get_task_mode_spec(QA_SHORT_TASK).synthesis_answer_guidance
    schema_instruction = (
        'Return a JSON array where each element is an object with keys "question" '
        'and "answer". '
    )
    task_instruction = (
        f"Generate {pair_instruction} whose questions and answers are drawn "
        "ONLY from facts stated in the [Target passage] above. "
    )
    fallback_task_instruction = (
        f"Below is a passage. Generate {pair_instruction} "
        "based ONLY on the content of this passage. "
    )

    if surrounding_context:
        user_content = (
            "Below is a target passage, preceded by surrounding context for "
            "background understanding only.\n\n"
            f"[Background context]\n{surrounding_context}\n\n"
            f"[Target passage]\n{chunk_text}\n\n"
            + task_instruction
            + answer_guidance
            + schema_instruction
            + "Do NOT include any explanation or text outside the JSON array."
        )
    else:
        user_content = (
            fallback_task_instruction
            + answer_guidance
            + schema_instruction
            + "Do NOT include any explanation or text outside the JSON array.\n\n"
            f"Passage:\n{chunk_text}"
        )

    messages = [{"role": "user", "content": user_content}]
    apply_kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if _supports_kwarg(tokenizer, "enable_thinking"):
        apply_kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(**apply_kwargs)


def generate_batch(
    llm,
    prompts: list[str],
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
) -> list[str]:
    """Run vLLM batch generation and return raw text outputs."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return [out.outputs[0].text for out in outputs]


def _validated_qa_items(parsed: Any) -> list[dict] | None:
    """Return validated question-answer items from a parsed JSON array."""
    if not isinstance(parsed, list):
        return None
    items: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            continue
        question = item.get("question")
        answer = item.get("answer")
        if (
            isinstance(question, str)
            and question.strip()
            and isinstance(answer, str)
            and answer.strip()
        ):
            items.append({"question": question.strip(), "answer": answer.strip()})
    return items


def _loads_qa_items(text: str) -> list[dict] | None:
    """Parse text as JSON and return validated QA items on success."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _validated_qa_items(parsed)


def _candidate_json_repairs(text: str) -> list[str]:
    """Return small, conservative repair variants for nearly-complete JSON arrays."""
    stripped = text.rstrip()
    if not stripped:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        if candidate and candidate not in seen and candidate != text:
            seen.add(candidate)
            candidates.append(candidate)

    if stripped.endswith(","):
        stripped = stripped[:-1].rstrip()

    _add(stripped + "]")
    _add(stripped + "}]")
    _add(stripped + "\"}]")
    return candidates


def parse_output(raw_text: str) -> list[dict]:
    """Parse vLLM output as a JSON array of question-answer objects."""
    text = raw_text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    bracket_idx = text.find("[")
    if bracket_idx != -1:
        text = text[bracket_idx:]

    parsed_items = _loads_qa_items(text)
    if parsed_items is not None:
        return parsed_items

    for repaired_text in _candidate_json_repairs(text):
        parsed_items = _loads_qa_items(repaired_text)
        if parsed_items is not None:
            log.debug("Recovered QA output via conservative JSON repair.")
            return parsed_items

    log.debug("Failed to parse QA output: %s", raw_text[:200])
    return []
