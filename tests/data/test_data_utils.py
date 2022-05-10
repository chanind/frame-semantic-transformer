from __future__ import annotations

from frame_semantic_transformer.data.data_utils import standardize_punct


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