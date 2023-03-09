from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frame_types import Frame
    from .loaders.loader import InferenceLoader


class LoaderDataCache:
    """
    Helper class which wraps a InferenceLoader and performs cached data lookups for performance
    """

    loader: InferenceLoader

    def __init__(self, loader: InferenceLoader):
        self.loader = loader

    def setup(self) -> None:
        """
        Perform any setup required, e.g. downloading needed data
        """
        self.loader.setup()

    @lru_cache(1)
    def get_frames_by_name(self) -> dict[str, Frame]:
        """
        cached fast-lookup helper for frame names -> frames for faster eval
        """
        results: dict[str, Frame] = {}
        for frame in self.loader.load_frames():
            results[normalize_name(frame.name)] = frame
        return results

    @lru_cache(1)
    def get_frame_element_name_loopkup(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for frame in self.get_frames_by_name().values():
            for element in frame.core_elements:
                results[normalize_name(element)] = element
            for element in frame.non_core_elements:
                results[normalize_name(element)] = element
        return results

    def get_frame(self, name: str) -> Frame:
        """
        Get a frame by name
        """
        return self.get_frames_by_name()[normalize_name(name)]

    def is_valid_frame(self, name: str) -> bool:
        """
        Check if a frame name is valid
        """
        return normalize_name(name) in self.get_frames_by_name()

    def standardize_element_name(self, name: str) -> str | None:
        """
        Standardize a frame element name
        """
        norm_name = normalize_name(name)
        if norm_name not in self.get_frame_element_name_loopkup():
            return None if self.loader.strict_frame_elements() else name
        return self.get_frame_element_name_loopkup()[norm_name]

    @lru_cache(1)
    def get_lexical_unit_bigram_to_frame_lookup_map(self) -> dict[str, list[str]]:
        """
        Return a mapping of lexical unit bigrams to the list of frames they are associated with
        """
        uniq_lookup_map: dict[str, set[str]] = defaultdict(set)
        for frame in self.get_frames_by_name().values():
            for lu in frame.lexical_units:
                parts = lu.split()
                lu_bigrams: list[str] = []
                prev_part = None
                for part in parts:
                    # also key this as a mongram if there's only 1 element or the word is rare enough
                    if len(parts) == 1 or self.loader.prioritize_lexical_unit(part):
                        for norm_part in self._normalize_lexical_unit_ngram([part]):
                            lu_bigrams.append(norm_part)
                    if prev_part is not None:
                        for norm_parts in self._normalize_lexical_unit_ngram(
                            [prev_part, part]
                        ):
                            lu_bigrams.append(norm_parts)
                    prev_part = part

                for bigram in lu_bigrams:
                    uniq_lookup_map[bigram].add(frame.name)
        sorted_lookup_map: dict[str, list[str]] = {}
        for lu_bigram, frames in uniq_lookup_map.items():
            sorted_lookup_map[lu_bigram] = sorted(list(frames))
        return sorted_lookup_map

    def get_possible_frames_for_trigger_bigrams(
        self, bigrams: list[list[str]]
    ) -> list[str]:
        possible_frames = []
        lookup_map = self.get_lexical_unit_bigram_to_frame_lookup_map()
        for bigram in bigrams:
            # sorted here just to get a consistent ordering
            for normalized_bigram in sorted(self._normalize_lexical_unit_ngram(bigram)):
                if normalized_bigram in lookup_map:
                    bigram_frames = lookup_map[normalized_bigram]
                    possible_frames += bigram_frames
        # remove duplicates, while preserving order
        # https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set/53657523#53657523
        return list(dict.fromkeys(possible_frames))

    def _normalize_lexical_unit_ngram(self, ngram: list[str]) -> set[str]:
        norm_toks = [
            setify(self.loader.normalize_lexical_unit_text(tok)) for tok in ngram
        ]
        return {"_".join(combo) for combo in product(*norm_toks)}


def normalize_name(name: str) -> str:
    """
    Normalize a frame or element name to be lowercase and without underscores
    """
    return name.lower().replace("_", "")


def setify(input: str | set[str]) -> set[str]:
    """
    Convert a string or set to a set
    """
    if isinstance(input, str):
        return {input}
    return input
