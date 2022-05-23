from __future__ import annotations
from functools import lru_cache
from typing import Any, Sequence, Mapping
import nltk

from nltk.corpus import framenet as fn


class InvalidFrameError(Exception):
    pass


def ensure_framenet_downloaded() -> None:
    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        nltk.download("framenet_v17")


def is_valid_frame(frame: str) -> bool:
    return frame in get_all_valid_frame_names()


def get_core_frame_elements(frame: str) -> list[str]:
    if not is_valid_frame(frame):
        raise InvalidFrameError(frame)
    return get_frame_elements_map_by_core_type()[frame]["core"]


def get_non_core_frame_elements(frame: str) -> list[str]:
    if not is_valid_frame(frame):
        raise InvalidFrameError(frame)
    return get_frame_elements_map_by_core_type()[frame]["noncore"]


@lru_cache(1)
def get_frame_elements_map_by_core_type() -> dict[str, dict[str, list[str]]]:
    """
    fast-lookup helper for frames -> frame elements for faster eval
    """
    results: dict[str, dict[str, list[str]]] = {}
    for frame in fn.frames():
        results[frame.name] = {"core": [], "noncore": []}
        for element_name, element in frame.FE.items():
            element_type = "core" if element.coreType == "Core" else "noncore"
            results[frame.name][element_type].append(element_name)
    return results


@lru_cache(1)
def get_all_valid_frame_names() -> set[str]:
    return {frame.name for frame in fn.frames()}


def get_lexical_units() -> Sequence[Mapping[str, Any]]:
    return fn.lus()


def get_fulltext_docs() -> Sequence[Mapping[str, Any]]:
    return fn.docs()
