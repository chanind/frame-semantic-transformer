from __future__ import annotations
import re
from typing import Iterator, Sequence, TypeVar
from difflib import SequenceMatcher, Match
from torch import Tensor

from transformers import T5TokenizerFast

from frame_semantic_transformer.constants import PADDING_LABEL_ID

T = TypeVar("T")


def chunk_list(lst: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def standardize_punct(sent: str) -> str:
    """
    Try to standardize things like "He 's a man" -> "He's a man"
    """
    updated_sent = T5TokenizerFast.clean_up_tokenization(sent)
    # remove space before punct
    updated_sent = re.sub(r"([a-zA-Z0-9])\s+(\*?[.',:?])", r"\1\2", updated_sent)
    # remove repeated *'s
    updated_sent = re.sub(r"\*+", "*", updated_sent)
    # fix spaces in contractions
    updated_sent = re.sub(r"([a-zA-Z0-9])\s+(\*?n't)", r"\1\2", updated_sent)
    # remove ``
    updated_sent = re.sub(r"\s*``\s*", " ", updated_sent)
    # replace "*n't" with "n*'t, for the tokenizer"
    updated_sent = re.sub(r"\*n't", "n*'t", updated_sent)
    # put a space between * and letter chars, since it seems to work better with the tokenizer
    updated_sent = re.sub(r"\*([a-zA-Z0-9])", r"* \1", updated_sent)

    return updated_sent.strip()


def marked_string_to_locs(origin_text: str, marked_text: str) -> list[int]:
    """
    Take a sentence like "He went to the store" and also the output of the trigger identification task, which
    will be a string like "He * went to the * store". We return the character indicies of the words in the
    original sentence that correspond to the marked words in the second sentence. In the example, the marked
    words are "went" and "store", and their indicies in the original sentence are 3 and 15, so the list
    [3, 15] is returned.

    The transformer that generates the marked text is trained to reproduce the original sentence, but there
    is nothing that guarantees that it will do so, and in some cases the transformer will even be inherently
    unable to reproduce the original sentence, such as when the original sentence contains substrings that
    the transformer's tokenizer parses to <unk>.

    Therefore this function does not assume that the marked sentence will be identical to the original sentence,
    modulo the trigger marks. Rather the trigger locations are transferred from the marked sentence to the original
    sentence via an alignment between the two sentences.
    """
    locs: list[int] = []

    # collapse the space between each "*" and the word it marks so that there are not extra spaces during alignment
    marked_text = marked_text.replace("* ", "*")

    # align the output of the trigger identification task with the original sentence
    matcher = SequenceMatcher(a=origin_text, b=marked_text, autojunk=False)
    matches = matcher.get_matching_blocks()

    # discard the ending null match since there shouldn't be a "*" after the last word, and, if there is one, we
    # don't want to record it as a valid trigger location
    matches = matches[:-1]

    # add a starting null match so that we can recognize a "*" marking the first word, if there happens to be one
    matches.insert(0, Match(0, 0, 0))

    # any one-character-wide gaps between matches that contain "*" we record as trigger locations
    for match, next_match in zip(matches[:-1], matches[1:]):
        if (
            match.a + match.size == next_match.a
            and match.b + match.size + len("*") == next_match.b
            and "*" == marked_text[match.b + match.size]
        ):
            locs.append(next_match.a)

    return locs


def trim_batch(
    input_ids: Tensor, attention_mask: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Helper to trim a batch of inputs / labels to strip padding down to the length of the longest item in the batch
    This helps the model run faster by avoiding needing to generate up to 512 characters of padding when the meaningful
    content is much shorter.
    """
    longest_inputs = int(attention_mask.sum(dim=1).max().item())
    longest_labels = int((labels != PADDING_LABEL_ID).sum(dim=1).max().item())

    truncated_input_ids = input_ids[:, :longest_inputs].contiguous()
    truncated_attention_mask = attention_mask[:, :longest_inputs].contiguous()
    truncated_labels = labels[:, :longest_labels].contiguous()

    return (truncated_input_ids, truncated_attention_mask, truncated_labels)
