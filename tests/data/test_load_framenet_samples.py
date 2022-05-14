from __future__ import annotations
from syrupy.assertion import SnapshotAssertion
from frame_semantic_transformer.data.framenet import get_fulltext_docs

from frame_semantic_transformer.data.load_framenet_samples import (
    load_sesame_dev_samples,
    load_sesame_test_samples,
    load_sesame_train_samples,
    parse_samples_from_fulltext_doc,
)


def test_load_sesame_test_samples() -> None:
    samples = load_sesame_test_samples()
    assert len(samples) == 13510


def test_load_sesame_dev_samples() -> None:
    samples = load_sesame_dev_samples()
    assert len(samples) == 4613


def test_load_sesame_train_samples() -> None:
    samples = load_sesame_train_samples()
    trigger_id_samples = [
        sample
        for sample in samples
        if sample.get_task_name() == "trigger_identification"
    ]
    frame_id_samples = [
        sample for sample in samples if sample.get_task_name() == "frame_classification"
    ]
    assert len(trigger_id_samples) == 3425
    assert len(frame_id_samples) == 20597
    assert len(samples) == 40233


def test_parse_samples_from_fulltext_doc(snapshot: SnapshotAssertion) -> None:
    doc = get_fulltext_docs()[17]  # random document
    samples = parse_samples_from_fulltext_doc(doc)
    assert samples == snapshot
