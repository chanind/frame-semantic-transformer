from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from .FrameClassificationTask import FrameClassificationTask
from .TaskSample import TaskSample


@dataclass
class FrameClassificationSample(TaskSample):
    task: FrameClassificationTask
    frame: str

    # -- input / target for training --

    def get_target(self) -> str:
        return self.frame

    @staticmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str], target: str, _input: str
    ) -> tuple[float, float, float]:
        prediction = FrameClassificationTask.parse_output(prediction_outputs)
        if prediction and prediction == target:
            return (1.0, 0.0, 0.0)
        else:
            # sesame treats any non-correct frame as both a false pos and false neg
            # https://github.com/swabhs/open-sesame/blob/master/sesame/evaluation.py#L67
            return (0.0, 1.0, 1.0)
