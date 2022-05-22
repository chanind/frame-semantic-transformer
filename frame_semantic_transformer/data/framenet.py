from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
from nltk.stem import PorterStemmer
from typing import Any, Sequence, Mapping
import nltk

from nltk.corpus import framenet as fn


stemmer = PorterStemmer()


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


def get_possible_frames_for_lexical_unit(lu: str) -> list[str]:
    # TODO: can make this smarter, especially for lus like "up" which can have
    # tons of possible matches. Ideally use bigrams as well
    lookup_map = get_lexical_unit_to_frame_lookup_map()
    return lookup_map.get(normalize_lexical_unit_text(lu), [])


@lru_cache(1)
def get_lexical_unit_to_frame_lookup_map() -> dict[str, list[str]]:
    uniq_lookup_map = defaultdict(set)
    for lu in fn.lus():
        parts = lu.name.split()
        for part in parts:
            normalized_part = normalize_lexical_unit_text(part)
            uniq_lookup_map[normalized_part].add(lu.frame.name)
    sorted_lookup_map: dict[str, list[str]] = {}
    for lu, frames in uniq_lookup_map.items():
        sorted_lookup_map[lu] = sorted(list(frames))
    return sorted_lookup_map


def normalize_lexical_unit_text(lu: str) -> str:
    normalized_lu = lu.lower()
    normalized_lu = re.sub(r"\.[a-zA-Z]+$", "", normalized_lu)
    normalized_lu = re.sub(r"[^a-z0-9 ]", "", normalized_lu)
    return stemmer.stem(normalized_lu.strip())


def get_fulltext_docs() -> Sequence[Mapping[str, Any]]:
    return fn.docs()
