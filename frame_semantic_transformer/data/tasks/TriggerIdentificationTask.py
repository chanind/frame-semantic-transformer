from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

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
    def parse_output(prediction_outputs: Sequence[str]) -> str:
        return prediction_outputs[0]
