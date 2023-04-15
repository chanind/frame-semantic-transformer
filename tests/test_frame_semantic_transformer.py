from syrupy.assertion import SnapshotAssertion

from frame_semantic_transformer import FrameSemanticTransformer


transformer = FrameSemanticTransformer("small")


def test_basic_detect_frames_functionality(snapshot: SnapshotAssertion) -> None:
    result = transformer.detect_frames(
        "I'm getting quite hungry, but I can wait a bit longer."
    )
    assert result.sentence == "I'm getting quite hungry, but I can wait a bit longer."
    assert result == snapshot


def test_problematic_sentence() -> None:
    result = transformer.detect_frames("two cars parked on the sidewalk on the street")
    assert result.sentence == "two cars parked on the sidewalk on the street"
    assert len(result.trigger_locations) > 0
    assert len(result.frames) > 0


def test_sentence_with_unk_chars() -> None:
    result = transformer.detect_frames(
        "The people you refer to (<PERSON>, <PERSON>, <PERSON>) were never involved."
    )
    assert (
        result.sentence
        == "The people you refer to (<PERSON>, <PERSON>, <PERSON>) were never involved."
    )
    assert len(result.trigger_locations) > 0
    assert len(result.frames) > 0


def test_no_results() -> None:
    result = transformer.detect_frames("nope")
    assert result.sentence == "nope"
    assert len(result.trigger_locations) == 0
    assert len(result.frames) == 0


def test_basic_detect_frames_bulk() -> None:
    sentences = [
        "I'm getting quite hungry, but I can wait a bit longer.",
        "The chef gave the food to the customer.",
        "The hallway smelt of boiled cabbage and old rag mats.",
    ]
    results = transformer.detect_frames_bulk(sentences)
    assert len(results) == 3
    for result, sentence in zip(results, sentences):
        assert result == transformer.detect_frames(sentence)
        assert result.sentence == sentence
        assert len(result.trigger_locations) > 0
        assert len(result.frames) > 0
