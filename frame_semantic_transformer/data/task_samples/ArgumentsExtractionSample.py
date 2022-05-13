from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from frame_semantic_transformer.data.task_samples.TaskSample import TaskSample


@dataclass
class ArgumentsExtractionSample(TaskSample):
    text: str
    trigger_loc: tuple[int, int]
    frame: str
    frame_element_locs: list[tuple[int, int, str]]

    # -- input / target / eval for training --

    def get_task_name(self) -> str:
        return "args_extraction"

    def get_input(self) -> str:
        return f"ARGS {self.frame}: {self.trigger_labeled_text}"

    def get_target(self) -> str:
        return " | ".join(
            [f"{element} = {text}" for element, text in self.frame_elements]
        )

    @staticmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str], target: str
    ) -> tuple[int, int, int]:
        # TODO: improve evaluation
        if prediction_outputs[0] == target:
            return (1, 0, 0)
        else:
            return (0, 1, 0)

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

    @property
    def frame_elements(self) -> list[tuple[str, str]]:
        return [
            (element, self.text[loc_start:loc_end])
            for (loc_start, loc_end, element) in self.frame_element_locs
        ]
