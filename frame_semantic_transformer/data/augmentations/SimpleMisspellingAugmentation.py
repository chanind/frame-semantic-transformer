from __future__ import annotations

import random
import string
from frame_semantic_transformer.data.augmentations.modification_helpers import (
    modify_text_without_changing_length,
)

from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation, ProbabilityType


class SimpleMisspellingAugmentation(DataAugmentation):

    max_misspellings_per_sentence: int
    min_misspellings_per_sentence: int

    def __init__(
        self,
        probability: ProbabilityType,
        max_misspellings_per_sentence: int = 10,
        min_misspellings_per_sentence: int = 1,
    ):
        super().__init__(probability)
        self.max_misspellings_per_sentence = max_misspellings_per_sentence
        self.min_misspellings_per_sentence = min_misspellings_per_sentence

    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        def misspell_cb(sentence: str) -> str:
            num_mispellings = random.randint(
                self.min_misspellings_per_sentence, self.max_misspellings_per_sentence
            )
            new_sentence = sentence
            for _ in range(num_mispellings):
                index = random.randint(0, len(sentence) - 1)
                char = sentence[index]
                new_char = char
                if char.isupper():
                    new_char = random.choice(string.ascii_uppercase)
                elif char.islower():
                    new_char = random.choice(string.ascii_lowercase)
                elif char.isdigit():
                    new_char = random.choice(string.digits)
                new_sentence = (
                    new_sentence[:index] + new_char + new_sentence[index + 1 :]
                )
            return new_sentence

        return modify_text_without_changing_length(task_sample, misspell_cb)
