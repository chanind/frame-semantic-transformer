from frame_semantic_transformer import FrameSemanticTransformer


def test_basic_detect_frames_functionality() -> None:
    transformer = FrameSemanticTransformer("small")
    result = transformer.detect_frames(
        "I'm getting quite hungry, but I can wait a bit longer."
    )
    assert result.sentence == "I'm getting quite hungry, but I can wait a bit longer."
    assert len(result.trigger_locations) > 0
    assert len(result.frames) > 0
