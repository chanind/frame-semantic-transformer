from frame_semantic_transformer.data.tasks.FrameClassificationTask import (
    FrameClassificationTask,
)


def test_trigger_bigrams() -> None:
    task = FrameClassificationTask(
        text="Your contribution to Goodwill will mean more than you may know .",
        trigger_loc=5,
    )

    assert task.trigger_bigrams == [
        ["Your", "contribution"],
        ["contribution", "to"],
        ["contribution"],
    ]
