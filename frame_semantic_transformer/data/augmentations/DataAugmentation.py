from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from random import uniform

if TYPE_CHECKING:
    from frame_semantic_transformer.data.tasks.TaskSample import TaskSample


class DataAugmentation(ABC):
    """
    Base class for data augmentations on training data
    """

    probability: float

    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, task_sample: TaskSample) -> TaskSample:
        """
        randomly apply this augmentation in proportion to self.probability
        """
        rand_val = uniform(0, 1.0)
        if rand_val > self.probability:
            return task_sample
        return self.apply_augmentation(task_sample)

    @abstractmethod
    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        """
        Main logic for subclasses to implement
        """
        pass
