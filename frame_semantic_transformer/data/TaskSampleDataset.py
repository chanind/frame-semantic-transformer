from __future__ import annotations
from collections import defaultdict
import random
from typing import Any, Sequence
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from frame_semantic_transformer.data.task_samples.TaskSample import TaskSample


MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512


class TaskSampleDataset(Dataset[Any]):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    samples: Sequence[TaskSample]

    def __init__(
        self,
        samples: Sequence[TaskSample],
        tokenizer: T5Tokenizer,
        balance_tasks: bool = False,
        seed: int = 42,
    ):
        samples_to_parse = samples
        if balance_tasks:
            samples_to_parse = balance_tasks_by_type(samples, seed)
        input_ids, attention_mask, labels = parse_samples(samples_to_parse, tokenizer)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.samples = samples

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def balance_tasks_by_type(
    samples: Sequence[TaskSample], seed: int
) -> Sequence[TaskSample]:
    """
    try to force an approximate balance of task types by repeating tasks of uncommon types
    """
    counts_by_type: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts_by_type[sample.get_task_name()] += 1
    max_task_count = max(counts_by_type.values())
    balanced_samples: list[TaskSample] = []
    for sample in samples:
        sample_ratio = int(max_task_count / counts_by_type[sample.get_task_name()])
        # duplicate each sample in proportion to how few tasks of this type are in the original mix
        for _ in range(sample_ratio):
            balanced_samples.append(sample)
    random.Random(seed).shuffle(balanced_samples)
    return balanced_samples


def parse_samples(
    samples: Sequence[TaskSample], tokenizer: T5Tokenizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_sequences: list[str] = []
    output_sequences: list[str] = []
    for sample in samples:
        input_sequences.append(sample.get_input())
        output_sequences.append(sample.get_target())

    input_encoding = tokenizer(
        input_sequences,
        padding="longest",
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = (
        input_encoding.input_ids,
        input_encoding.attention_mask,
    )
    output_encoding = tokenizer(
        output_sequences,
        padding="longest",
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    labels = torch.tensor(output_encoding.input_ids)
    labels[labels == tokenizer.pad_token_id] = -100

    return (input_ids, attention_mask, labels)
