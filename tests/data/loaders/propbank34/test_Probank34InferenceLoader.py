from __future__ import annotations

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.loaders.propbank34 import Propbank34InferenceLoader

pb_loader = Propbank34InferenceLoader()
pb_loader_cache = LoaderDataCache(pb_loader)


def test_get_lexical_unit_bigram_to_frame_lookup_map() -> None:
    lookup_map = pb_loader_cache.get_lexical_unit_bigram_to_frame_lookup_map()
    assert len(lookup_map) > 5000

    for _lu, frames in lookup_map.items():
        assert len(frames) < 22
