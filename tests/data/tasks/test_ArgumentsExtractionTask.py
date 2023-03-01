from __future__ import annotations

from frame_semantic_transformer.data.tasks.ArgumentsExtractionTask import (
    split_output_fe_spans,
)


def test_split_output_fe_spans() -> None:
    target = "Donor = Your | Recipient = to Goodwill"
    split_output_fe_spans(target) == [
        ("Donor", "Your"),
        ("Recipient", "to Goodwill"),
    ]


def test_split_output_fe_spans_fixes_broken_contractions() -> None:
    target = "Theme = benefits that many entry - level jobs don 't include"
    split_output_fe_spans(target) == [
        ("Donor", "Your"),
        ("Theme", "benefits that many entry - level jobs don't include"),
    ]
