from __future__ import annotations
import random

from frame_semantic_transformer.data.augmentations.modification_helpers import (
    splice_text,
)
from frame_semantic_transformer.data.tasks import TaskSample
from .DataAugmentation import DataAugmentation
from .modification_helpers.get_sample_text import get_sample_text


LATEX_QUOTES = ["``", "''"]
STANDARD_QUOTE = '"'
ALL_QUOTES = LATEX_QUOTES + [STANDARD_QUOTE]


class DoubleQuotesAugmentation(DataAugmentation):
    def apply_augmentation(self, task_sample: TaskSample) -> TaskSample:
        sample_text = get_sample_text(task_sample)

        # if standard quotes are used, convert to latex quotes, and vice versa
        to_latex = STANDARD_QUOTE in sample_text
        from_quotes = [STANDARD_QUOTE] if to_latex else LATEX_QUOTES
        to_quotes = LATEX_QUOTES if to_latex else [STANDARD_QUOTE]

        updated_sample = task_sample
        while count_instances(sample_text, from_quotes) > 0:
            quote, start_loc = find_first_instance(sample_text, from_quotes)
            try:
                updated_sample = splice_text(
                    updated_sample,
                    lambda _text, _critical_indices: (
                        start_loc,
                        len(quote),
                        random.choice(to_quotes),
                    ),
                )
                sample_text = get_sample_text(updated_sample)
            except ValueError:
                # The splice failed, so just return the sample
                return updated_sample

        return updated_sample


def count_instances(text: str, substrings: list[str]) -> int:
    return sum(text.count(substring) for substring in substrings)


def find_first_instance(text: str, substrings: list[str]) -> tuple[str, int]:
    """
    Find the first instance of any of the substrings in the text. Returns the substring and the
    start location of the substring.
    """
    for substring in substrings:
        start_loc = text.find(substring)
        if start_loc >= 0:
            return substring, start_loc
    raise ValueError(f"Could not find any of {substrings} in {text}")
