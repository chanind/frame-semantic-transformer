from __future__ import annotations
from syrupy.assertion import SnapshotAssertion
from frame_semantic_transformer.data.framenet import get_fulltext_docs

from frame_semantic_transformer.data.load_framenet_samples import (
    load_sesame_dev_samples,
    load_sesame_test_samples,
    parse_samples_from_fulltext_doc,
)


def test_load_sesame_test_samples() -> None:
    assert len(load_sesame_test_samples()) == 11604


def test_load_sesame_dev_samples() -> None:
    assert len(load_sesame_dev_samples()) == 3816


def test_parse_samples_from_fulltext_doc(snapshot: SnapshotAssertion) -> None:
    doc = get_fulltext_docs()[17]  # random document
    samples = parse_samples_from_fulltext_doc(doc)
    assert samples == snapshot
