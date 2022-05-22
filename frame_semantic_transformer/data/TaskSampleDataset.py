from __future__ import annotations
from collections import defaultdict
import random
from typing import Any, Callable, Optional, Sequence
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer
from frame_semantic_transformer.constants import MODEL_MAX_LENGTH, PADDING_LABEL_ID

from frame_semantic_transformer.data.augmentations import (
    LowercaseAugmentation,
    RemoveContractionsAugmentation,
    RemoveEndPunctuationAugmentation,
    chain_augmentations,
)
from frame_semantic_transformer.data.tasks import TaskSample


MAX_TARGET_LEN = 512


class TaskSampleDataset(Dataset[Any]):
    samples: Sequence[TaskSample]
    augmentation: Optional[Callable[[str, str], tuple[str, str]]] = None
    tokenizer: T5Tokenizer

    def __init__(
        self,
        samples: Sequence[TaskSample],
        tokenizer: T5Tokenizer,
        balance_tasks: bool = False,
        seed: int = 42,
        max_task_duplication_factor: int = 2,
        augment_data: bool = False,
    ):
        self.samples = samples
        if balance_tasks:
            self.samples = balance_tasks_by_type(
                samples, seed=seed, max_duplication_factor=max_task_duplication_factor
            )
        self.tokenizer = tokenizer
        if augment_data:
            self.augmentation = chain_augmentations(
                [
                    RemoveEndPunctuationAugmentation(0.3),
                    LowercaseAugmentation(0.2),
                    RemoveContractionsAugmentation(0.2),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        input_ids, attention_mask, labels = self.parse_sample(sample)

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": labels,
            "task": sample.get_task_name(),
        }

    def parse_sample(
        self, sample: TaskSample
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input = sample.get_input()
        target = sample.get_target()
        if self.augmentation:
            input, target = self.augmentation(input, target)

        input_encoding = self.tokenizer(
            input,
            padding="max_length",
            max_length=MODEL_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            input_encoding.input_ids,
            input_encoding.attention_mask,
        )
        output_encoding = self.tokenizer(
            target,
            padding="max_length",
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
        labels = torch.tensor(output_encoding.input_ids)
        labels[labels == self.tokenizer.pad_token_id] = PADDING_LABEL_ID

        return (input_ids, attention_mask, labels)


def balance_tasks_by_type(
    samples: Sequence[TaskSample],
    max_duplication_factor: int = 2,
    seed: int = 42,
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
        duplication_factor = int(
            max_task_count / counts_by_type[sample.get_task_name()]
        )
        # duplicate each sample in proportion to how few tasks of this type are in the original mix
        for _ in range(min(max_duplication_factor, duplication_factor)):
            balanced_samples.append(sample)
    random.Random(seed).shuffle(balanced_samples)
    return balanced_samples
