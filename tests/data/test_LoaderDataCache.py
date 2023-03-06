from __future__ import annotations

from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache


def test_get_lexical_unit_bigram_to_frame_lookup_map(
    loader_cache: LoaderDataCache,
) -> None:
    lookup_map = loader_cache.get_lexical_unit_bigram_to_frame_lookup_map()
    assert len(lookup_map) > 5000
    for frames in lookup_map.values():
        assert len(frames) < 20


def test_normalize_lexical_unit_ngram(loader_cache: LoaderDataCache) -> None:
    assert loader_cache._normalize_lexical_unit_ngram(["can't", "stop"]) == {
        "cant_stop"
    }
    assert loader_cache._normalize_lexical_unit_ngram(["he", "eats"]) == {"he_eat"}
    assert loader_cache._normalize_lexical_unit_ngram(["eats"]) == {"eat"}


def test_get_possible_frames_for_trigger_bigrams(loader_cache: LoaderDataCache) -> None:
    assert loader_cache.get_possible_frames_for_trigger_bigrams(
        [["can't", "help"], ["help", "it"], ["help"]]
    ) == ["Self_control", "Assistance"]
    assert loader_cache.get_possible_frames_for_trigger_bigrams(
        [["can't", "help"]]
    ) == ["Self_control"]
    assert loader_cache.get_possible_frames_for_trigger_bigrams(
        [["and", "staffed"], ["staffed", "by"], ["staffed"]]
    ) == ["Employing", "Working_a_post"]


def test_get_possible_frames_for_trigger_bigrams_stems_bigrams(
    loader_cache: LoaderDataCache,
) -> None:
    assert loader_cache.get_possible_frames_for_trigger_bigrams(
        [["can't", "helps"]]
    ) == ["Self_control"]
