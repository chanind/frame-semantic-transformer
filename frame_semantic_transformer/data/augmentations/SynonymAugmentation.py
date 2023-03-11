from __future__ import annotations

from nlpaug.augmenter.word import SynonymAug

from frame_semantic_transformer.data.augmentations.modification_helpers import (
    splice_text,
)
from frame_semantic_transformer.data.augmentations.modification_helpers.splice_text import (
    is_valid_splice,
)

from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation, ProbabilityType


class SynonymAugmentation(DataAugmentation):
    """
    Wrapper about nlpaug's SynonymAugmenter
    """

    augmenter: SynonymAug

    def __init__(self, probability: ProbabilityType):
        super().__init__(probability)
        self.augmenter = SynonymAug(aug_max=1, aug_min=1)
        self.augmenter.include_detail = True

    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        def splice_end_punct_cb(
            sentence: str, critical_indices: list[int]
        ) -> tuple[int, int, str] | None:
            _, changes = self.augmenter.augment(sentence)[0]
            if len(changes) == 0:
                return None
            start = changes[0]["orig_start_pos"]
            new_text = changes[0]["new_token"]
            del_len = len(changes[0]["orig_token"])
            if not is_valid_splice(start, del_len, critical_indices):
                return None
            return start, del_len, new_text

        return splice_text(task_sample, splice_end_punct_cb)
