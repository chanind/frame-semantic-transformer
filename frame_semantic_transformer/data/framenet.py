from __future__ import annotations
from functools import lru_cache
from typing import Any, Sequence, Mapping
import nltk

from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader.framenet import FramenetCorpusReader


class FullFramenetCorpusReader(FramenetCorpusReader):
    """
    Hacky class to get nltk to stop skipping "Problem" lus
    """

    _bad_statuses: list[str] = []


# copied from https://github.com/nltk/nltk/blob/34ee17b395b1bb18cf307bdafb3feea99bf54243/nltk/corpus/__init__.py#L152
# but using the modified corpus reader above, to avoid missing "Problem" lexical units
fn = LazyCorpusLoader(
    "framenet_v17",
    FullFramenetCorpusReader,
    [
        "frRelation.xml",
        "frameIndex.xml",
        "fulltextIndex.xml",
        "luIndex.xml",
        "semTypes.xml",
    ],
)


class InvalidFrameError(Exception):
    pass


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
    return {frame.name for frame in get_frames()}


def get_lexical_units() -> Sequence[Mapping[str, Any]]:
    return fn.lus()


def get_frames() -> Sequence[Mapping[str, Any]]:
    return fn.frames()


def get_fulltext_docs() -> Sequence[Mapping[str, Any]]:
    return fn.docs()
