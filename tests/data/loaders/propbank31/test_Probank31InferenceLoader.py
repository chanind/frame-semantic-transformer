from __future__ import annotations

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.loaders.propbank31 import Propbank31InferenceLoader

pb_loader = Propbank31InferenceLoader()
pb_loader_cache = LoaderDataCache(pb_loader)


def test_get_lexical_unit_bigram_to_frame_lookup_map() -> None:
    lookup_map = pb_loader_cache.get_lexical_unit_bigram_to_frame_lookup_map()
    assert len(lookup_map) > 5000

    for lu, frames in lookup_map.items():
        if len(frames) >= 21:
            print(lu, frames)
        assert len(frames) < 21
