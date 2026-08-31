"""Label-building helpers for the mixed SFT sub-pipeline."""

from __future__ import annotations


def _offset_tokenize(
    tokenizer,
    text: str,
    *,
    max_seq_length: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])[:max_seq_length]
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][:max_seq_length]]
    return input_ids, offsets


def _fallback_prefix_mask(
    tokenizer,
    full_text: str,
    prompt_prefix: str,
    *,
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)[:max_seq_length]
    prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    prefix_len = min(len(prefix_ids), len(input_ids))
    continuation_mask = [0] * len(input_ids)
    for idx in range(prefix_len, len(input_ids)):
        continuation_mask[idx] = 1
    return input_ids, continuation_mask


def _fallback_span_mask(
    tokenizer,
    full_text: str,
    spans: list[tuple[int, int]],
    *,
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)[:max_seq_length]
    train_mask = [0] * len(input_ids)
    for start, end in spans:
        token_start = len(tokenizer.encode(full_text[:start], add_special_tokens=False))
        token_end = len(tokenizer.encode(full_text[:end], add_special_tokens=False))
        token_start = min(max(0, token_start), len(input_ids))
        token_end = min(max(token_start, token_end), len(input_ids))
        for idx in range(token_start, token_end):
            train_mask[idx] = 1
    return input_ids, train_mask


def tokenize_continuation_example(
    tokenizer,
    full_text: str,
    prompt_prefix: str,
    *,
    max_seq_length: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize a prompt+completion string and mark continuation positions."""

    try:
        input_ids, offsets = _offset_tokenize(
            tokenizer,
            full_text,
            max_seq_length=max_seq_length,
        )
        boundary = len(prompt_prefix)
        continuation_mask = [1 if start >= boundary else 0 for start, _ in offsets]
    except Exception:
        input_ids, continuation_mask = _fallback_prefix_mask(
            tokenizer,
            full_text,
            prompt_prefix,
            max_seq_length=max_seq_length,
        )

    if not input_ids:
        return None
    if sum(continuation_mask) <= 0:
        return None
    return input_ids, continuation_mask


def tokenize_spans_example(
    tokenizer,
    full_text: str,
    spans: list[tuple[int, int]],
    *,
    max_seq_length: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize text and mark the requested character spans as trainable."""

    normalized_spans = [
        (max(0, int(start)), min(len(full_text), int(end)))
        for start, end in spans
        if int(end) > int(start)
    ]
    if not normalized_spans:
        return None

    try:
        input_ids, offsets = _offset_tokenize(
            tokenizer,
            full_text,
            max_seq_length=max_seq_length,
        )
        train_mask = []
        for token_start, token_end in offsets:
            keep = any(
                token_end > span_start and token_start < span_end
                for span_start, span_end in normalized_spans
            )
            train_mask.append(1 if keep else 0)
    except Exception:
        input_ids, train_mask = _fallback_span_mask(
            tokenizer,
            full_text,
            normalized_spans,
            max_seq_length=max_seq_length,
        )

    if not input_ids:
        return None
    if sum(train_mask) <= 0:
        return None
    return input_ids, train_mask


def build_continuation_labels(
    input_ids: list[int],
    continuation_mask: list[int],
) -> list[int]:
    """Convert a continuation mask into causal-LM labels."""

    return [token_id if keep else -100 for token_id, keep in zip(input_ids, continuation_mask)]
