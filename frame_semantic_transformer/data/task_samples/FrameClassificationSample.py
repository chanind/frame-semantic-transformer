from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from frame_semantic_transformer.data.data_utils import standardize_punct
from frame_semantic_transformer.data.framenet import is_valid_frame

from frame_semantic_transformer.data.task_samples.TaskSample import TaskSample


@dataclass
class FrameClassificationSample(TaskSample):
    text: str
    trigger_loc: tuple[int, int]
    frame: str

    # -- input / target for training --

    def get_task_name(self) -> str:
        return "frame_classification"

    def get_input(self) -> str:
        return f"FRAME: {self.trigger_labeled_text}"

    def get_target(self) -> str:
        return self.frame

    @staticmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str], target: str, _input: str
    ) -> tuple[float, float, float]:
        valid_predictions = [
            pred for pred in prediction_outputs if is_valid_frame(pred)
        ]
        if len(valid_predictions) > 0 and valid_predictions[0] == target:
            return (1.0, 0.0, 0.0)
        else:
            # sesame treats any non-correct frame as both a false pos and false neg
            # https://github.com/swabhs/open-sesame/blob/master/sesame/evaluation.py#L67
            return (0.0, 1.0, 1.0)

    # -- helper properties --

    @property
    def trigger(self) -> str:
        return self.text[self.trigger_loc[0] : self.trigger_loc[1]]

    @property
    def trigger_labeled_text(self) -> str:
        pre_span = self.text[0 : self.trigger_loc[0]]
        post_span = self.text[self.trigger_loc[1] :]
        # TODO: handle these special chars better
        return standardize_punct(f"{pre_span}*{self.trigger}{post_span}")
