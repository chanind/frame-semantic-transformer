from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Sequence
from frame_semantic_transformer.data.framenet import (
    get_non_core_frame_elements,
)

from .ArgumentsExtractionTask import ArgumentsExtractionTask, split_output_fe_spans
from .TaskSample import TaskSample


@dataclass
class ArgumentsExtractionSample(TaskSample):
    task: ArgumentsExtractionTask
    frame_element_locs: list[tuple[int, int, str]]

    # -- input / target / eval for training --

    def get_target(self) -> str:
        return " | ".join(
            [f"{element} = {text}" for element, text in self.frame_elements]
        )

    @staticmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str],
        target: str,
        input: str,
    ) -> tuple[float, float, float]:
        # based on argid eval in sesame (labeled_eval)
        # from https://github.com/swabhs/open-sesame/blob/master/sesame/evaluation.py

        frame = re.sub(r"\s+\|.*$", "", input.replace("ARGS ", "")).strip()

        true_pos = 0.0
        false_pos = 0.0
        false_neg = 0.0
        target_spans = split_output_fe_spans(target)
        prediction_spans = ArgumentsExtractionTask.parse_output(prediction_outputs)

        for target_span in target_spans:
            score = get_eval_score(frame, target_span[0])
            if target_span in prediction_spans:
                true_pos += score
            else:
                false_neg += score

        for prediction_span in prediction_spans:
            if prediction_span not in target_spans:
                score = get_eval_score(frame, prediction_span[0])
                false_pos += score

        return (true_pos, false_pos, false_neg)

    # -- helper properties --

    @property
    def frame_elements(self) -> list[tuple[str, str]]:
        return [
            (element, self.task.text[loc_start:loc_end])
            for (loc_start, loc_end, element) in self.frame_element_locs
        ]


def get_eval_score(frame: str, element: str) -> float:
    if element in get_non_core_frame_elements(frame):
        return 0.5
    return 1.0
