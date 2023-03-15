from __future__ import annotations
from typing import Callable, Sequence

from frame_semantic_transformer.data.tasks import TaskSample

from .DataAugmentation import DataAugmentation


def chain_augmentations(
    augmentations: Sequence[DataAugmentation],
) -> Callable[[TaskSample], TaskSample]:
    def chained_augmentation(input: TaskSample) -> TaskSample:
        chained_input = input
        for augmentation in augmentations:
            chained_input = augmentation(chained_input)
        return chained_input

    return chained_augmentation
