from __future__ import annotations

from nlpaug.augmenter.char import KeyboardAug

from frame_semantic_transformer.data.augmentations.modification_helpers import (
    modify_text_without_changing_length,
)
from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation


class KeyboardAugmentation(DataAugmentation):
    """
    Wrapper about nlpaug's KeyboardAugmenter
    Attempts to make spelling mistakes similar to what a user might make
    """

    augmenter: KeyboardAug

    def __init__(self, probability: float):
        super().__init__(probability)
        self.augmenter = KeyboardAug(include_special_char=False)
        self.augmenter.include_detail = True

    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        def augment_sent(sentence: str) -> str:
            new_sentence, changes = self.augmenter.augment(sentence)[0]
            # sometimes this augmenter changes token lengths, which we don't want
            # just skip the changes if that happens
            for change in changes:
                if len(change["orig_token"]) != len(change["new_token"]):
                    return sentence
            return new_sentence

        return modify_text_without_changing_length(task_sample, augment_sent)
