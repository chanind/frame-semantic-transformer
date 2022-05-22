from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from frame_semantic_transformer.data.data_utils import standardize_punct
from frame_semantic_transformer.data.framenet import (
    get_possible_frames_for_lexical_unit,
    is_valid_frame,
)

from .Task import Task


@dataclass
class FrameClassificationTask(Task):
    text: str
    trigger_loc: int

    # -- input / target for training --

    @staticmethod
    def get_task_name() -> str:
        return "frame_classification"

    def get_input(self) -> str:
        potential_frames = get_possible_frames_for_lexical_unit(self.trigger)
        return f"FRAME {' '.join(potential_frames)} : {self.trigger_labeled_text}"

    @staticmethod
    def parse_output(prediction_outputs: Sequence[str]) -> str | None:
        for pred in prediction_outputs:
            if is_valid_frame(pred):
                return pred
        return None

    # -- helper properties --

    @property
    def trigger(self) -> str:
        return self.text[self.trigger_loc :].split()[0]

    @property
    def trigger_labeled_text(self) -> str:
        pre_span = self.text[0 : self.trigger_loc]
        post_span = self.text[self.trigger_loc :]
        # TODO: handle these special chars better
        return standardize_punct(f"{pre_span}*{post_span}")
