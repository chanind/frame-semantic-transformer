from __future__ import annotations
from abc import ABC, abstractmethod
from random import uniform


class DataAugmentation(ABC):
    """
    Base class for data augmentations on training data
    """

    probability: float

    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, input: str, output: str) -> tuple[str, str]:
        """
        randomly apply this augmentation in proportion to self.probability
        """
        rand_val = uniform(0, 1.0)
        if rand_val > self.probability:
            return (input, output)
        return self.apply_augmentation(input, output)

    @abstractmethod
    def apply_augmentation(self, input: str, output: str) -> tuple[str, str]:
        """
        Main logic for subclasses to implement
        """
        pass
