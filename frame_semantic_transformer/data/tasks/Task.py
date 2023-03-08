from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache


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
    def parse_output(
        prediction_outputs: Sequence[str],
        loader_cache: LoaderDataCache,
    ) -> Any:
        pass

    @abstractmethod
    def get_input(self) -> str:
        pass
