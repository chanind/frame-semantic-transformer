from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Sequence
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from frame_semantic_transformer.data.frame_types import FrameElementAnnotation

from .ArgumentsExtractionTask import ArgumentsExtractionTask, split_output_fe_spans
from .TaskSample import TaskSample


@dataclass
class ArgumentsExtractionSample(TaskSample):
    task: ArgumentsExtractionTask
    frame_elements: list[FrameElementAnnotation]

    # -- input / target / eval for training --

    def get_target(self) -> str:
        return " | ".join(
            [f"{element} = {text}" for element, text in self.labeled_frame_elements]
        )

    @staticmethod
    def evaluate_prediction(
        prediction_outputs: Sequence[str],
        target: str,
        input: str,
        loader_cache: LoaderDataCache,
    ) -> tuple[float, float, float]:
        # based on argid eval in sesame (labeled_eval)
        # from https://github.com/swabhs/open-sesame/blob/master/sesame/evaluation.py

        frame = re.sub(r"\s+\|.*$", "", input.replace("ARGS ", "")).strip()

        true_pos = 0.0
        false_pos = 0.0
        false_neg = 0.0
        target_spans = split_output_fe_spans(target)
        prediction_spans = ArgumentsExtractionTask.parse_output(
            prediction_outputs, loader_cache
        )

        for target_span in target_spans:
            score = get_eval_score(frame, target_span[0], loader_cache)
            if target_span in prediction_spans:
                true_pos += score
            else:
                false_neg += score

        for prediction_span in prediction_spans:
            if prediction_span not in target_spans:
                score = get_eval_score(frame, prediction_span[0], loader_cache)
                false_pos += score

        return (true_pos, false_pos, false_neg)

    # -- helper properties --

    @property
    def labeled_frame_elements(self) -> list[tuple[str, str]]:
        """
        Return a list of tuples of the form (frame_element, text) for all frame elements, instead of indices
        """
        return [
            (fe.name, self.task.text[fe.start_loc : fe.end_loc])
            for fe in self.frame_elements
        ]


def get_eval_score(frame: str, element: str, loader_cache: LoaderDataCache) -> float:
    if element in loader_cache.get_frame(frame).non_core_elements:
        return 0.5
    return 1.0
