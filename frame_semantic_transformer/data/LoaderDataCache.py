from __future__ import annotations
from collections import defaultdict
from functools import lru_cache

from .loaders.loader import InferenceLoader
from .frame_types import Frame


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
            results[frame.name] = frame
        return results

    def get_frame(self, name: str) -> Frame:
        """
        Get a frame by name
        """
        return self.get_frames_by_name()[name]

    def is_valid_frame(self, name: str) -> bool:
        """
        Check if a frame name is valid
        """
        return name in self.get_frames_by_name()

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
                        lu_bigrams.append(self._normalize_lexical_unit_ngram([part]))
                    if prev_part is not None:
                        lu_bigrams.append(
                            self._normalize_lexical_unit_ngram([prev_part, part])
                        )
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
            normalized_bigram = self._normalize_lexical_unit_ngram(bigram)
            if normalized_bigram in lookup_map:
                bigram_frames = lookup_map[normalized_bigram]
                possible_frames += bigram_frames
        # remove duplicates, while preserving order
        # https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set/53657523#53657523
        return list(dict.fromkeys(possible_frames))

    def _normalize_lexical_unit_ngram(self, ngram: list[str]) -> str:
        return "_".join([self.loader.normalize_lexical_unit_text(tok) for tok in ngram])
