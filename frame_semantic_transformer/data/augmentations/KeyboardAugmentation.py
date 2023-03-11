from __future__ import annotations

from nlpaug.augmenter.char import KeyboardAug

from frame_semantic_transformer.data.augmentations.modification_helpers import (
    modify_text_without_changing_length,
)
from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation, ProbabilityType


class KeyboardAugmentation(DataAugmentation):
    """
    Wrapper about nlpaug's KeyboardAugmenter
    Attempts to make spelling mistakes similar to what a user might make
    """

    augmenter: KeyboardAug

    def __init__(self, probability: ProbabilityType):
        super().__init__(probability)
        self.augmenter = KeyboardAug(
            include_special_char=False, aug_char_p=0.1, aug_word_p=0.1
        )
        self.augmenter.include_detail = True

    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        def augment_sent(sentence: str) -> str:
            # this augmentation removes spaces around punctuation, so just manually do the changes
            _, changes = self.augmenter.augment(sentence)[0]
            new_sentence = sentence
            for change in changes:
                # sometimes this augmenter changes token lengths, which we don't want
                # just skip the changes if that happens
                if len(change["orig_token"]) != len(change["new_token"]):
                    return new_sentence
                if change["orig_start_pos"] != change["new_start_pos"]:
                    return new_sentence
                start_pos = change["orig_start_pos"]
                end_pos = change["orig_start_pos"] + len(change["orig_token"])
                new_sentence = (
                    new_sentence[:start_pos]
                    + change["new_token"]
                    + new_sentence[end_pos:]
                )
            return new_sentence

        return modify_text_without_changing_length(task_sample, augment_sent)
