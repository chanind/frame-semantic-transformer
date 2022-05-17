from __future__ import annotations
import re
from typing import Iterator, Sequence, TypeVar

from transformers import T5Tokenizer

T = TypeVar("T")


def chunk_list(lst: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def standardize_punct(sent: str) -> str:
    """
    Try to standardize things like "He 's a man" -> "He's a man"
    """
    updated_sent = T5Tokenizer.clean_up_tokenization(sent)
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


def marked_string_to_locs(
    text: str, symbol: str = "*", remove_spaces: bool = True
) -> tuple[str, list[int]]:
    """
    Take a string like "He * went to the * store" and return the indices of the tagged words,
    in this case "went" and "store", and remove the tags (in this case the *'s)
    """
    output_str = ""
    remaining_str = text
    locs: list[int] = []
    symbol_index = remaining_str.find("*")

    while symbol_index != -1:
        locs.append(symbol_index + len(output_str))
        output_str += remaining_str[:symbol_index]
        remaining_str = remaining_str[symbol_index + len(symbol) :]
        if remove_spaces:
            remaining_str = remaining_str.strip()
        symbol_index = remaining_str.find("*")
    output_str += remaining_str
    return output_str, locs
