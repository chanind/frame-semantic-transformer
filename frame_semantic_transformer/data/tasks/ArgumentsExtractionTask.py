from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
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
        prediction_outputs: Sequence[str], _loader_cache: LoaderDataCache
    ) -> list[tuple[str, str]]:
        return split_output_fe_spans(prediction_outputs[0])

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
    for span in output.split("|"):
        parts = span.strip().split("=")
        if len(parts) == 1:
            outputs.append((parts[0].strip(), "N/A"))
        else:
            outputs.append((parts[0].strip(), parts[1].strip()))
    return outputs
