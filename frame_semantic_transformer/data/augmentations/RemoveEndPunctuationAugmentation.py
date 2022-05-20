from __future__ import annotations
from .DataAugmentation import DataAugmentation
import re

REMOVE_END_PUNCT_RE = r"\s*[.?!]\s*$"


class RemoveEndPunctuationAugmentation(DataAugmentation):
    def apply_augmentation(self, input: str, output: str) -> tuple[str, str]:
        return (
            re.sub(REMOVE_END_PUNCT_RE, "", input),
            re.sub(REMOVE_END_PUNCT_RE, "", output),
        )
