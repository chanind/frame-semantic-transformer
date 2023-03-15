from __future__ import annotations
import random
import string

from frame_semantic_transformer.data.augmentations.modification_helpers import (
    splice_text,
)
from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation, ProbabilityType
from .modification_helpers.get_sample_text import get_sample_text


class StripPunctuationAugmentation(DataAugmentation):

    max_to_remove: int
    min_to_remove: int

    def __init__(
        self,
        probability: ProbabilityType,
        max_to_remove: int = 5,
        min_to_remove: int = 1,
    ):
        self.max_to_remove = max_to_remove
        self.min_to_remove = min_to_remove
        super().__init__(probability)

    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        sample_text = get_sample_text(task_sample)
        punctuation_indices = find_punctuation_indices(sample_text)
        if len(punctuation_indices) == 0:
            return task_sample

        updated_sample = task_sample
        for _ in range(random.randint(self.min_to_remove, self.max_to_remove)):
            if len(punctuation_indices) == 0:
                break
            punctuation_index = random.choice(punctuation_indices)
            punctuation_indices.remove(punctuation_index)
            punctuation_indices = [
                i - 1 if i > punctuation_index else i for i in punctuation_indices
            ]
            try:
                updated_sample = splice_text(
                    updated_sample,
                    lambda _text, _critical_indices: (
                        punctuation_index,
                        1,
                        "",
                    ),
                )
            except ValueError:
                # The splice failed, so just return the sample
                return updated_sample
        return updated_sample


def find_punctuation_indices(text: str) -> list[int]:
    """
    Find the indices of all punctuation in the text.
    """
    # TODO: This would be more efficient with a regex
    return [i for i, char in enumerate(text) if char in string.punctuation]
