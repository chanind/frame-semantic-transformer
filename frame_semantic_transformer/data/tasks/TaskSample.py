from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence

from frame_semantic_transformer.data.tasks.Task import Task


class TaskSample(ABC):
    """
    Abstract interface for all Task Samples
    """

    task: Task

    def get_task_name(self) -> str:
        return self.task.get_task_name()

    def get_input(self) -> str:
        return self.task.get_input()

    @abstractmethod
    def get_target(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str], target: str, input: str
    ) -> tuple[float, float, float]:
        "return a tuple indicating the number of true positives, false positives, and false negatives in the prediction"
        pass
