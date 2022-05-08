from __future__ import annotations
from functools import lru_cache
from typing import Any, Sequence, Mapping
import nltk

from nltk.corpus import framenet as fn


def ensure_framenet_downloaded() -> None:
    nltk.download("framenet_v17")


def is_valid_frame(frame: str) -> bool:
    return frame in get_all_valid_frame_names()


@lru_cache(1)
def get_all_valid_frame_names() -> set[str]:
    return {frame.name for frame in fn.frames()}


def get_fulltext_docs() -> Sequence[Mapping[str, Any]]:
    return fn.docs()
