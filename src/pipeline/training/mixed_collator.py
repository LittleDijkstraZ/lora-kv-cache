"""Batch collation for tokenized mixed-SFT examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MixedDataCollator:
    """Pad tokenized examples with per-sample labels."""

    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_length - seq_len
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
