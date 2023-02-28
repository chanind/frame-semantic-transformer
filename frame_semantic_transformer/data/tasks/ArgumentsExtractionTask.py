from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from transformers import T5TokenizerFast

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.data_utils import standardize_punct


from .Task import Task


@dataclass
class ArgumentsExtractionTask(Task):
    text: str
    trigger_loc: int
    frame: str
    loader_cache: LoaderDataCache

    @staticmethod
    def get_task_name() -> str:
        return "args_extraction"

    def get_input(self) -> str:
        frame = self.loader_cache.get_frame(self.frame)
        core_elements = frame.core_elements
        non_core_elements = frame.non_core_elements
        # put core elements in front
        elements = [*core_elements, *non_core_elements]
        return f"ARGS {self.frame} | {' '.join(elements)} : {self.trigger_labeled_text}"

    @staticmethod
    def parse_output(
        prediction_outputs: Sequence[str], loader_cache: LoaderDataCache
    ) -> list[tuple[str, str]]:
        raw_elements = split_output_fe_spans(prediction_outputs[0])
        processed_elements = []
        for element, text in raw_elements:
            fixed_element = loader_cache.standardize_element_name(element)
            if fixed_element is not None:
                processed_elements.append((fixed_element, text))
        return processed_elements

    @property
    def trigger_labeled_text(self) -> str:
        pre_span = self.text[0 : self.trigger_loc]
        post_span = self.text[self.trigger_loc :]
        # TODO: handle these special chars better
        return standardize_punct(f"{pre_span}*{post_span}")


def split_output_fe_spans(output: str) -> list[tuple[str, str]]:
    """
    Split an output like "Agent = He | Destination = to the store" into a list of elements and values, like:
    [("Agent", "He"), ("Destination", "to the store")]
    """
    outputs: list[tuple[str, str]] = []
    for span in T5TokenizerFast.clean_up_tokenization(output).split("|"):
        parts = span.strip().split("=")
        if len(parts) == 1:
            # invalid output - just skip this
            continue
        else:
            outputs.append(
                (
                    parts[0].strip(),
                    parts[1].strip(),
                )
            )
    return outputs
