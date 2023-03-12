from __future__ import annotations

from typing import Callable, Union
from abc import ABC, abstractmethod
from random import uniform

from frame_semantic_transformer.data.tasks.TaskSample import TaskSample


ProbabilityType = Union[float, Callable[[TaskSample], float]]


class DataAugmentation(ABC):
    """
    Base class for data augmentations on training data
    """

    probability: ProbabilityType

    def __init__(self, probability: ProbabilityType):
        self.probability = probability

    def __call__(self, task_sample: TaskSample) -> TaskSample:
        """
        randomly apply this augmentation in proportion to self.probability
        """
        rand_val = uniform(0, 1.0)
        if rand_val > self.get_probability(task_sample):
            return task_sample
        return self.apply_augmentation(task_sample)

    def get_probability(self, task_sample: TaskSample) -> float:
        if callable(self.probability):
            return self.probability(task_sample)
        return self.probability

    @abstractmethod
    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        """
        Main logic for subclasses to implement
        """
        pass
