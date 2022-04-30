from __future__ import annotations
from syrupy.assertion import SnapshotAssertion
from nltk.corpus import framenet as fn

from frame_semantic_transformer.data.SampleSentence import (
    SampleSentence,
    parse_samples_from_fulltext_doc,
    parse_samples_from_lexical_unit,
)


def test_parse_samples_from_lexical_unit(snapshot: SnapshotAssertion) -> None:
    lu = fn.lu(6403)  # repair.v

    samples = parse_samples_from_lexical_unit(lu)

    assert len(samples) == 9
    assert (
        samples[0].text
        == "The word is that the usurper , or those acting in his name , have sent out a call for all Scots lords and landed men to repair there , to Annan , to do homage to him . "
    )
    assert (
        samples[0].trigger_labeled_text
        == "The word is that the usurper , or those acting in his name , have sent out a call for all Scots lords and landed men to * repair * there , to Annan , to do homage to him . "
    )

    assert samples[0].trigger == "repair"
    assert samples[0].frame_elements == [
        ("Self_mover", "all Scots lords and landed men"),
        ("Goal", "there , to Annan ,"),
        ("Purpose", "to do homage to him"),
    ]
    assert (
        samples[0].frame_elements_str
        == "Self_mover = all Scots lords and landed men | Goal = there , to Annan , | Purpose = to do homage to him"
    )
    assert samples == snapshot


def test_parse_samples_from_fulltext_doc(snapshot: SnapshotAssertion) -> None:
    doc = fn.docs()[17]  # random document
    samples = parse_samples_from_fulltext_doc(doc)
    assert samples == snapshot
