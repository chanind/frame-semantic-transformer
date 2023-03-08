from __future__ import annotations
import re
from frame_semantic_transformer.data.augmentations.modification_helpers import (
    splice_text,
)
from frame_semantic_transformer.data.augmentations.modification_helpers.splice_text import (
    is_valid_splice,
)

from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation

REMOVE_END_PUNCT_RE = r"\s*[.?!]\s*$"


class RemoveEndPunctuationAugmentation(DataAugmentation):
    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        def splice_end_punct_cb(
            sentence: str, critical_indices: list[int]
        ) -> tuple[int, int, str] | None:
            match = re.search(REMOVE_END_PUNCT_RE, sentence)
            if match is None:
                return None
            start, end = match.span()
            del_len = end - start
            if not is_valid_splice(start, del_len, critical_indices):
                return None
            return start, del_len, ""

        return splice_text(task_sample, splice_end_punct_cb)
