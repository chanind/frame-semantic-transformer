from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.data_utils import standardize_punct


from .Task import Task


@dataclass
class FrameClassificationTask(Task):
    text: str
    trigger_loc: int
    loader_cache: LoaderDataCache

    # -- input / target for training --

    @staticmethod
    def get_task_name() -> str:
        return "frame_classification"

    def get_input(self) -> str:
        potential_frames = self.loader_cache.get_possible_frames_for_trigger_bigrams(
            self.trigger_bigrams
        )
        return f"FRAME {' '.join(potential_frames)} : {self.trigger_labeled_text}"

    @staticmethod
    def parse_output(
        prediction_outputs: Sequence[str], loader_cache: LoaderDataCache
    ) -> str | None:
        for pred in prediction_outputs:
            if loader_cache.is_valid_frame(pred):
                # fix any capitalization differences between pred and the frame name
                return loader_cache.get_frame(pred).name
        return None

    # -- helper properties --

    @property
    def trigger_bigrams(self) -> list[list[str]]:
        """
        return bigrams of the trigger, trigger + next word, and prev word + trigger
        """
        pre_trigger_tokens = self.text[: self.trigger_loc].split()
        trigger_and_after_tokens = self.text[self.trigger_loc :].split()
        trigger = trigger_and_after_tokens[0]
        post_trigger_tokens = trigger_and_after_tokens[1:]
        bigrams: list[list[str]] = []
        if len(pre_trigger_tokens) > 0:
            bigrams.append([pre_trigger_tokens[-1], trigger])
        if len(post_trigger_tokens) > 0:
            bigrams.append([trigger, post_trigger_tokens[0]])
        # add the monogram last
        bigrams.append([trigger])
        return bigrams

    @property
    def trigger_labeled_text(self) -> str:
        pre_span = self.text[0 : self.trigger_loc]
        post_span = self.text[self.trigger_loc :]
        # TODO: handle these special chars better
        return standardize_punct(f"{pre_span}*{post_span}")
