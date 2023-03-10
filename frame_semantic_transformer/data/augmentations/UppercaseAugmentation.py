from __future__ import annotations
from frame_semantic_transformer.data.augmentations.modification_helpers import (
    modify_text_without_changing_length,
)

from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation


class UppercaseAugmentation(DataAugmentation):
    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        def safe_uppercase(text: str) -> str:
            new_text = text.upper()
            # it turns out the some characters, like "ﬁ", become 2 chars when uppercased
            # just check to make sure we're not in that case here
            if len(new_text) != len(text):
                return text
            return new_text

        return modify_text_without_changing_length(task_sample, safe_uppercase)
