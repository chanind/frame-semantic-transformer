from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Sequence
from frame_semantic_transformer.data.framenet import (
    get_core_frame_elements,
    get_non_core_frame_elements,
)

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
        core_elements = get_core_frame_elements(self.frame)
        non_core_elements = get_non_core_frame_elements(self.frame)
        # put core elements in front
        elements = [*core_elements, *non_core_elements]
        return f"ARGS {self.frame} | {' '.join(elements)} : {self.trigger_labeled_text}"

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
        prediction_spans = split_output_fe_spans(prediction_outputs[0])

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


def get_eval_score(frame: str, element: str) -> float:
    if element in get_non_core_frame_elements(frame):
        return 0.5
    return 1.0


def split_output_fe_spans(output: str) -> list[tuple[str, str]]:
    """
    Split an output like "Agent = He | Destination = to the store" into a list of elements and values, like:
    [("Agent", "He"), ("Destination", "to the store")]
    """
    outputs: list[tuple[str, str]] = []
    for span in output.split("|"):
        parts = span.strip().split("=")
        if len(parts) == 1:
            outputs.append((parts[0].strip(), "XXX"))
        else:
            outputs.append((parts[0].strip(), parts[1].strip()))
    return outputs
