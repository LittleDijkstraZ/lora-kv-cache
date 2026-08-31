"""Input formatters for the four training formats reported in the paper."""

from __future__ import annotations

import inspect
from typing import Tuple

from transformers import PreTrainedTokenizerBase

from src.pipeline.task_modes import get_qa_boxed_instruction, normalize_task_mode


RAW_FORMAT = "raw"
CHUNK_NEXT_PROMPT_FORMAT = "chunk_next_prompt"
QA_ONLY_FORMAT = "qa_only"
CTX_QA_FORMAT = "ctx_qa"
SUPPORTED_INPUT_FORMATS = (
    RAW_FORMAT,
    CHUNK_NEXT_PROMPT_FORMAT,
    QA_ONLY_FORMAT,
    CTX_QA_FORMAT,
)

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


def _ensure_chat_template(tokenizer: PreTrainedTokenizerBase) -> None:
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = _DEFAULT_LLAMA3_CHAT_TEMPLATE


def _supports_apply_chat_kwarg(
    tokenizer: PreTrainedTokenizerBase, kwarg_name: str
) -> bool:
    try:
        signature = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError):
        return False
    return kwarg_name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    _ensure_chat_template(tokenizer)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if _supports_apply_chat_kwarg(tokenizer, "enable_thinking"):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def _append_eos(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    append_eos: bool,
) -> str:
    if not append_eos:
        return text
    if not tokenizer.eos_token:
        raise ValueError("append_eos=True requires tokenizer.eos_token")
    return text if text.endswith(tokenizer.eos_token) else text + tokenizer.eos_token


def format_record_for_sft(
    record: dict,
    tokenizer: PreTrainedTokenizerBase,
    input_format: str,
    *,
    task_mode: str | None = None,
    append_eos: bool = False,
) -> Tuple[str, str]:
    """Format a synthetic record as QA-only or context-plus-QA training."""

    normalized = input_format.strip().lower()
    if normalized not in {QA_ONLY_FORMAT, CTX_QA_FORMAT}:
        raise ValueError(f"Unsupported synthetic-record format: {input_format!r}")
    task = normalize_task_mode(task_mode or record.get("task", "qa_short"))
    instruction = get_qa_boxed_instruction(task)
    question = str(record["question"])
    answer = str(record["answer"])
    if normalized == CTX_QA_FORMAT:
        context = str(record.get("chunk_text") or record.get("context", ""))
        user_content = f"Here is a context: \n\n{context}\nQuestion: {question} " + instruction
    else:
        user_content = f"Question: {question} " + instruction

    answer_prefix = "The answer is \\boxed{"
    user_messages = [{"role": "user", "content": user_content}]
    full_messages = user_messages + [
        {"role": "assistant", "content": answer_prefix + answer + "}."}
    ]
    full_text = _apply_chat_template(
        tokenizer, full_messages, add_generation_prompt=False
    )
    prompt_prefix = (
        _apply_chat_template(tokenizer, user_messages, add_generation_prompt=True)
        + answer_prefix
    )
    return _append_eos(full_text, tokenizer, append_eos), prompt_prefix


def format_text_for_ntp(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    input_format: str = RAW_FORMAT,
) -> str:
    """Return raw text for the raw-context training format."""

    if input_format.strip().lower() != RAW_FORMAT:
        raise ValueError(f"Unsupported raw-context format: {input_format!r}")
    return text


def format_next_chunks_for_sft(
    *,
    current_text: str,
    next_text: str,
    tokenizer: PreTrainedTokenizerBase,
    input_format: str,
    append_eos: bool = False,
) -> Tuple[str, str]:
    """Format current-to-next chunk continuation training."""

    current = current_text.strip()
    target = next_text.strip()
    if not current or not target:
        return "", ""
    normalized = input_format.strip().lower()
    if normalized == RAW_FORMAT:
        prompt_prefix = current + "\n\n"
        return _append_eos(prompt_prefix + target, tokenizer, append_eos), prompt_prefix
    if normalized != CHUNK_NEXT_PROMPT_FORMAT:
        raise ValueError(f"Unsupported chunk-continuation format: {input_format!r}")

    user_messages = [
        {
            "role": "user",
            "content": (
                "Here is a chunk from a document:\n\n"
                f"{current}\n\nContinue the document with the next text."
            ),
        }
    ]
    full_messages = user_messages + [{"role": "assistant", "content": target}]
    full_text = _apply_chat_template(
        tokenizer, full_messages, add_generation_prompt=False
    )
    prompt_prefix = _apply_chat_template(
        tokenizer, user_messages, add_generation_prompt=True
    )
    return _append_eos(full_text, tokenizer, append_eos), prompt_prefix
