from __future__ import annotations
from frame_semantic_transformer.data.get_possible_frames_for_trigger_bigrams import (
    get_lexical_unit_bigram_to_frame_lookup_map,
    normalize_lexical_unit_ngram,
    get_possible_frames_for_trigger_bigrams,
)


def test_get_lexical_unit_bigram_to_frame_lookup_map() -> None:
    lookup_map = get_lexical_unit_bigram_to_frame_lookup_map()
    assert len(lookup_map) > 5000
    for frames in lookup_map.values():
        assert len(frames) < 20


def test_normalize_lexical_unit_ngram() -> None:
    assert normalize_lexical_unit_ngram(["can't", "stop"]) == "cant_stop"
    assert normalize_lexical_unit_ngram(["he", "eats"]) == "he_eat"
    assert normalize_lexical_unit_ngram(["eats"]) == "eat"


def test_get_possible_frames_for_trigger_bigrams() -> None:
    assert get_possible_frames_for_trigger_bigrams(
        [["can't", "help"], ["help", "it"], ["help"]]
    ) == ["Self_control", "Assistance"]
    assert get_possible_frames_for_trigger_bigrams([["can't", "help"]]) == [
        "Self_control"
    ]


def test_get_possible_frames_for_trigger_bigrams_stems_bigrams() -> None:
    assert get_possible_frames_for_trigger_bigrams([["can't", "helps"]]) == [
        "Self_control"
    ]
