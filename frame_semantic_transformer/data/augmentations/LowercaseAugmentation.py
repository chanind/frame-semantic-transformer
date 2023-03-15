from __future__ import annotations
from frame_semantic_transformer.data.augmentations.modification_helpers import (
    modify_text_without_changing_length,
)

from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation


class LowercaseAugmentation(DataAugmentation):
    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        return modify_text_without_changing_length(task_sample, str.lower)
