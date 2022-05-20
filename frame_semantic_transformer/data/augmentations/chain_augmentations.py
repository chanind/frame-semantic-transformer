from __future__ import annotations
from typing import Callable, Sequence

from .DataAugmentation import DataAugmentation


def chain_augmentations(
    augmentations: Sequence[DataAugmentation],
) -> Callable[[str, str], tuple[str, str]]:
    def chained_augmentation(input: str, output: str) -> tuple[str, str]:
        chained_input = input
        chained_output = output
        for augmentation in augmentations:
            chained_input, chained_output = augmentation(chained_input, chained_output)
        return chained_input, chained_output

    return chained_augmentation
