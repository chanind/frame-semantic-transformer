from __future__ import annotations
from .DataAugmentation import DataAugmentation
import re


def remove_contractions(text: str) -> str:
    new_text = text.replace("won't", "will not")
    new_text = new_text.replace("can't", "cannot")
    new_text = re.sub(r"n't(\b)", r" not\1", new_text)
    new_text = re.sub(r"'ll(\b)", r" will\1", new_text)
    new_text = re.sub(r"'m(\b)", r" am\1", new_text)
    new_text = re.sub(r"'re(\b)", r" are\1", new_text)
    new_text = re.sub(r"'ve(\b)", r" have\1", new_text)
    return new_text


class RemoveContractionsAugmentation(DataAugmentation):
    def apply_augmentation(self, input: str, output: str) -> tuple[str, str]:
        if "*'" in input or "*'" in output or "*n'" in input or "*n'" in output:
            return (input, output)

        return (
            remove_contractions(input),
            remove_contractions(output),
        )
