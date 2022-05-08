from __future__ import annotations
from abc import ABC, abstractmethod


class TaskSample(ABC):
    """
    Abstract interface for all Task Samples
    """

    @abstractmethod
    def get_task_name(self) -> str:
        pass

    @abstractmethod
    def get_input(self) -> str:
        pass

    @abstractmethod
    def get_target(self) -> str:
        pass

    @abstractmethod
    def evaluate_prediction(self, prediction: str) -> tuple[int, int, int]:
        "return a tuple indicating the number of true positives, false positives, and false negatives in the prediction"
        pass
