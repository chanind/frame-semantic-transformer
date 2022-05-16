from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence


class Task(ABC):
    """
    Abstract interface for all Tasks
    """

    @staticmethod
    @abstractmethod
    def get_task_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def parse_output(prediction_outputs: Sequence[str]) -> Any:
        pass

    @abstractmethod
    def get_input(self) -> str:
        pass
