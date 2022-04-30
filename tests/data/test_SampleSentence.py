from syrupy.assertion import SnapshotAssertion
from nltk.corpus import framenet as fn

from frame_semantic_transformer.data.SampleSentence import parse_samples_from_exemplars


def test_parse_samples_from_exemplars(snapshot: SnapshotAssertion) -> None:
    lu = fn.lu(6403)  # repair.v

    samples = parse_samples_from_exemplars(lu.exemplars)

    assert len(samples) == 9
    assert (
        samples[0].text
        == "The word is that the usurper , or those acting in his name , have sent out a call for all Scots lords and landed men to repair there , to Annan , to do homage to him . "
    )
    assert samples[0].trigger == "repair"
    assert samples[0].frame_elements == [
        ("Self_mover", "all Scots lords and landed men"),
        ("Goal", "there , to Annan ,"),
        ("Purpose", "to do homage to him"),
    ]
    assert samples == snapshot
