from __future__ import annotations
from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def chunk_list(lst: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
