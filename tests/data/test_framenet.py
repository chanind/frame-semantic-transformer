from __future__ import annotations
from frame_semantic_transformer.data.framenet import (
    get_lexical_unit_to_frame_lookup_map,
)


def test_get_lexical_unit_to_frame_lookup_map() -> None:
    lookup_map = get_lexical_unit_to_frame_lookup_map()
    assert len(lookup_map) > 5000
    for frames in lookup_map.values():
        assert len(frames) < 80
