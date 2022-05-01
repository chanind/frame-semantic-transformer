from __future__ import annotations
from typing import Sequence, TypeVar
from random import Random

T = TypeVar("T")


def shuffle_and_split(
    data: Sequence[T], train_ratio: float = 0.8, seed: int = 0
) -> tuple[Sequence[T], Sequence[T]]:
    random = Random(seed)
    shuffled = list(data)
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    return shuffled[:split_point], shuffled[split_point:]
