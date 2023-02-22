from __future__ import annotations

import pytest
import torch

from frame_semantic_transformer.data.data_utils import (
    marked_string_to_locs,
    standardize_punct,
    trim_batch,
)


def test_standardize_punct_removes_spaces_before_punctuation() -> None:
    original = "Old customs are still followed : Fate and luck are taken very seriously , and astrologers and fortune-tellers do a steady business ."
    expected = "Old customs are still followed: Fate and luck are taken very seriously, and astrologers and fortune-tellers do a steady business."
    assert standardize_punct(original) == expected


def test_standardize_punct_leaves_sentences_as_is_if_punct_is_correct() -> None:
    sent = "Old customs are still followed: Fate and luck are taken very seriously, and astrologers and fortune-tellers do a steady business."
    assert standardize_punct(sent) == sent


def test_standardize_punct_leaves_spaces_before_double_apostrophes() -> None:
    sent = "I really * like my * job. '' -- Sherry"
    assert standardize_punct(sent) == sent


def test_standardize_punct_keeps_asterix_before_apostrophes() -> None:
    original = "*Shopping *never *ends - *there *'s *always *another inviting *spot"
    expected = (
        "* Shopping * never * ends - * there*'s * always * another inviting * spot"
    )
    assert standardize_punct(original) == expected


def test_standardize_punct_removes_repeated_asterixes() -> None:
    original = "*Shopping **never *ends"
    expected = "* Shopping * never * ends"
    assert standardize_punct(original) == expected


def test_standardize_punct_undoes_spaces_in_contractions() -> None:
    original = "She did n't say so"
    expected = "She didn't say so"
    assert standardize_punct(original) == expected


def test_standardize_punct_asterix_in_contractions() -> None:
    original = "She did *n't say so"
    expected = "She didn*'t say so"
    assert standardize_punct(original) == expected


def test_standardize_punct_removes_weird_double_backticks() -> None:
    original = "`` I was *sad *when I *couldn't *go *to the snack bar to *buy a soda."
    expected = (
        "I was * sad * when I * couldn't * go * to the snack bar to * buy a soda."
    )
    assert standardize_punct(original) == expected


def test_standardize_punct_handles_question_marks() -> None:
    original = "Does Iran *intend to *become a Nuclear State ?"
    expected = "Does Iran * intend to * become a Nuclear State?"
    assert standardize_punct(original) == expected


def test_standardize_punct_removes_spaces_before_commas() -> None:
    original = "2- * Sheik of Albu'Ubaid ( Salah al-Dhari ) , who * slaughtered * thirty sheeps"
    expected = (
        "2- * Sheik of Albu'Ubaid ( Salah al-Dhari ), who * slaughtered * thirty sheeps"
    )
    assert standardize_punct(original) == expected


def test_standardize_punct_with_accented_letters() -> None:
    original = "C'est *très *bien."
    expected = "C'est * très * bien."
    assert standardize_punct(original) == expected


def test_standardize_punct_with_swedish() -> None:
    original = "Axel fick parkerade på Odengatan , detta gjorde honom pepp ."
    expected = "Axel fick parkerade på Odengatan, detta gjorde honom pepp."
    assert standardize_punct(original) == expected


@pytest.mark.parametrize(
    "input,expected",
    [
        ("Hi * there", ("Hi there", [3])),
        ("Hi there", ("Hi there", [])),
        (
            "Does Iran * intend to * become a Nuclear State?",
            ("Does Iran intend to become a Nuclear State?", [10, 20]),
        ),
        (
            "Does Iran * intend to *become a Nuclear State?",
            ("Does Iran intend to become a Nuclear State?", [10, 20]),
        ),
    ],
)
def test_marked_string_to_locs(input: str, expected: tuple[str, list[int]]) -> None:
    assert marked_string_to_locs(input) == expected


def test_trim_batch() -> None:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 2, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    labels = torch.tensor(
        [
            [1, 2, 3, -100, -100, -100],
            [17, -100, -100, -100, -100, -100],
            [2, 1, -100, -100, -100, -100],
        ]
    )
    trimmed_input_ids, trimmed_attention_mask, trimmed_labels = trim_batch(
        input_ids, attention_mask, labels
    )
    assert torch.equal(
        trimmed_input_ids,
        torch.tensor(
            [
                [1, 2, 3, 2, 0],
                [1, 2, 3, 4, 5],
                [2, 1, 0, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        trimmed_attention_mask,
        torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0],
            ]
        ),
    )
    assert torch.equal(
        trimmed_labels,
        torch.tensor(
            [
                [1, 2, 3],
                [17, -100, -100],
                [2, 1, -100],
            ]
        ),
    )
