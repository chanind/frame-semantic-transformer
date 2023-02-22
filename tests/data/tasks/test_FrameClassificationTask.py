from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.tasks.FrameClassificationTask import (
    FrameClassificationTask,
)


def test_trigger_bigrams(loader_cache: LoaderDataCache) -> None:
    task = FrameClassificationTask(
        text="Your contribution to Goodwill will mean more than you may know .",
        trigger_loc=5,
        loader_cache=loader_cache,
    )

    assert task.trigger_bigrams == [
        ["Your", "contribution"],
        ["contribution", "to"],
        ["contribution"],
    ]
