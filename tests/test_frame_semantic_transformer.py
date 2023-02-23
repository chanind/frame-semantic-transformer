from syrupy.assertion import SnapshotAssertion

from frame_semantic_transformer import FrameSemanticTransformer


def test_basic_detect_frames_functionality(snapshot: SnapshotAssertion) -> None:
    transformer = FrameSemanticTransformer("small")
    result = transformer.detect_frames(
        "I'm getting quite hungry, but I can wait a bit longer."
    )
    assert result.sentence == "I'm getting quite hungry, but I can wait a bit longer."
    assert result == snapshot


def test_basic_detect_frames_bulk() -> None:
    sentences = [
        "I'm getting quite hungry, but I can wait a bit longer.",
        "The chef gave the food to the customer.",
        "The hallway smelt of boiled cabbage and old rag mats.",
    ]
    transformer = FrameSemanticTransformer("small")
    results = transformer.detect_frames_bulk(sentences)
    assert len(results) == 3
    for result, sentence in zip(results, sentences):
        assert result == transformer.detect_frames(sentence)
        assert result.sentence == sentence
        assert len(result.trigger_locations) > 0
        assert len(result.frames) > 0
