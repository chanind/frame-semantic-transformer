from __future__ import annotations
from transformers import T5Tokenizer
from frame_semantic_transformer.data.TaskSampleDataset import (
    TaskSampleDataset,
    balance_tasks_by_type,
)
from frame_semantic_transformer.data.framenet import get_fulltext_docs

from frame_semantic_transformer.data.load_framenet_samples import (
    parse_samples_from_fulltext_doc,
)
from frame_semantic_transformer.data.task_samples.ArgumentsExtractionSample import (
    ArgumentsExtractionSample,
)
from frame_semantic_transformer.data.task_samples.FrameClassificationSample import (
    FrameClassificationSample,
)
from frame_semantic_transformer.data.task_samples.TriggerIdentificationSample import (
    TriggerIdentificationSample,
)


def test_TaskSampleDataset() -> None:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    doc = get_fulltext_docs()[1]
    # use the first 8 samples
    samples = parse_samples_from_fulltext_doc(doc)[0:8]

    dataset = TaskSampleDataset(samples, tokenizer)

    assert len(dataset) == 8
    assert len(dataset[0]["input_ids"]) == 55
    assert len(dataset[0]["attention_mask"]) == 55
    assert len(dataset[0]["labels"]) == 30


def test_balance_tasks_by_type() -> None:
    tasks = [
        ArgumentsExtractionSample("a1", (0, 2), "Greetings", []),
        ArgumentsExtractionSample("a2", (0, 2), "Greetings", []),
        ArgumentsExtractionSample("a3", (0, 2), "Greetings", []),
        ArgumentsExtractionSample("a4", (0, 2), "Greetings", []),
        ArgumentsExtractionSample("a5", (0, 2), "Greetings", []),
        FrameClassificationSample("f1", (0, 2), "Greetings"),
        FrameClassificationSample("f2", (0, 2), "Greetings"),
        TriggerIdentificationSample("t1", []),
    ]
    balanced_tasks = balance_tasks_by_type(tasks, 42)
    assert len(balanced_tasks) == 14  # 5 arg, 4 frame, 5 trigger
    assert (
        len([t for t in balanced_tasks if t.get_task_name() == "args_extraction"]) == 5
    )
    assert (
        len([t for t in balanced_tasks if t.get_task_name() == "frame_classification"])
        == 4
    )
    assert (
        len(
            [t for t in balanced_tasks if t.get_task_name() == "trigger_identification"]
        )
        == 5
    )
