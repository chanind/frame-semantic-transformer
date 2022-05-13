from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence


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

    @staticmethod
    @abstractmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str], target: str
    ) -> tuple[int, int, int]:
        "return a tuple indicating the number of true positives, false positives, and false negatives in the prediction"
        pass
