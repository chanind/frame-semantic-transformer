from __future__ import annotations
from dataclasses import dataclass

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

    def evaluate_prediction(self, prediction: str) -> tuple[int, int, int]:
        if prediction == self.frame:
            return (1, 0, 0)
        else:
            # sesame treats any non-correct frame as both a false pos and false neg
            # https://github.com/swabhs/open-sesame/blob/master/sesame/evaluation.py#L67
            return (0, 1, 1)

    # -- helper properties --

    @property
    def trigger(self) -> str:
        return self.text[self.trigger_loc[0] : self.trigger_loc[1]]

    @property
    def trigger_labeled_text(self) -> str:
        pre_span = self.text[0 : self.trigger_loc[0]]
        post_span = self.text[self.trigger_loc[1] :]
        # TODO: handle these special chars better
        return f"{pre_span}* {self.trigger} *{post_span}"
