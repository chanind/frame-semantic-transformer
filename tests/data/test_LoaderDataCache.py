from __future__ import annotations

import pytest

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
    assert loader_cache._normalize_lexical_unit_ngram(["he", "eats"]) == {
        "he_eat",
        "he_eats",
    }
    assert loader_cache._normalize_lexical_unit_ngram(["eats"]) == {"eat", "eats"}


@pytest.mark.parametrize(
    "ngrams,expected",
    [
        ([["can't", "help"], ["help", "it"], ["help"]], ["Self_control", "Assistance"]),
        ([["can't", "help"]], ["Self_control"]),
        (
            [["and", "staffed"], ["staffed", "by"], ["staffed"]],
            ["Employing", "Working_a_post"],
        ),
        (
            [["strongest"]],
            [
                "Chemical_potency",
                "Expertise",
                "Judgment_of_intensity",
                "Level_of_force_exertion",
                "Level_of_force_resistance",
                "Usefulness",
            ],
        ),
        (
            [["done"]],
            [
                "Activity_done_state",
                "Ingest_substance",
                "Intentionally_act",
                "Intentionally_affect",
                "Process_completed_state",
                "Sex",
                "Thriving",
                "Touring",
                "Dressing",
                "Giving",
            ],
        ),
    ],
)
def test_get_possible_frames_for_trigger_bigrams(
    ngrams: list[list[str]], expected: list[str], loader_cache: LoaderDataCache
) -> None:
    assert loader_cache.get_possible_frames_for_trigger_bigrams(ngrams) == expected


def test_get_possible_frames_for_trigger_bigrams_stems_bigrams(
    loader_cache: LoaderDataCache,
) -> None:
    assert loader_cache.get_possible_frames_for_trigger_bigrams(
        [["can't", "helps"]]
    ) == ["Self_control"]


@pytest.mark.parametrize(
    "ngrams,expected",
    [
        (
            [["use", "trying"], ["trying"], ["trying", "to"]],
            [
                "Attempt",
                "Attempt_means",
                "Operational_testing",
                "Tasting",
                "Trial",
                "Try_defendant",
                "Trying_out",
            ],
        ),
        (
            [["the lift"], ["lift"]],
            [
                "Body_movement",
                "Building_subparts",
                "Cause_change_of_position_on_a_scale",
                "Cause_motion",
                "Cause_to_end",
                "Connecting_architecture",
                "Theft",
            ],
        ),
    ],
)
def test_get_possible_frames_for_trigger_bigrams_paper_examples(
    ngrams: list[list[str]],
    expected: list[str],
    loader_cache: LoaderDataCache,
) -> None:
    assert loader_cache.get_possible_frames_for_trigger_bigrams(ngrams) == expected
