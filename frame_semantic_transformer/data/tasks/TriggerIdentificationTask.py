from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from .Task import Task


@dataclass
class TriggerIdentificationTask(Task):
    text: str

    # -- input / target for training --

    @staticmethod
    def get_task_name() -> str:
        return "trigger_identification"

    def get_input(self) -> str:
        return f"TRIGGER: {self.text}"

    @staticmethod
    def parse_output(
        prediction_outputs: Sequence[str], _loader_cache: LoaderDataCache
    ) -> str:
        return prediction_outputs[0]
