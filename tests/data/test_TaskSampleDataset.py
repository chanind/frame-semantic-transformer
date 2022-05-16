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
from frame_semantic_transformer.data.tasks.ArgumentsExtractionSample import (
    ArgumentsExtractionSample,
)
from frame_semantic_transformer.data.tasks.ArgumentsExtractionTask import (
    ArgumentsExtractionTask,
)
from frame_semantic_transformer.data.tasks.FrameClassificationSample import (
    FrameClassificationSample,
)
from frame_semantic_transformer.data.tasks.FrameClassificationTask import (
    FrameClassificationTask,
)
from frame_semantic_transformer.data.tasks.TriggerIdentificationSample import (
    TriggerIdentificationSample,
)
from frame_semantic_transformer.data.tasks.TriggerIdentificationTask import (
    TriggerIdentificationTask,
)


def test_TaskSampleDataset() -> None:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    doc = get_fulltext_docs()[1]
    # use the first 8 samples
    samples = parse_samples_from_fulltext_doc(doc)[0:8]

    dataset = TaskSampleDataset(samples, tokenizer)

    assert len(dataset) == 8
    assert len(dataset[0]["input_ids"]) == 99
    assert len(dataset[0]["attention_mask"]) == 99
    assert len(dataset[0]["labels"]) == 30


def test_balance_tasks_by_type() -> None:
    tasks = [
        ArgumentsExtractionSample(ArgumentsExtractionTask("a1", 0, "Greetings"), []),
        ArgumentsExtractionSample(ArgumentsExtractionTask("a2", 0, "Greetings"), []),
        ArgumentsExtractionSample(ArgumentsExtractionTask("a3", 0, "Greetings"), []),
        ArgumentsExtractionSample(ArgumentsExtractionTask("a4", 0, "Greetings"), []),
        ArgumentsExtractionSample(ArgumentsExtractionTask("a5", 0, "Greetings"), []),
        FrameClassificationSample(FrameClassificationTask("f1", 0), "Greetings"),
        FrameClassificationSample(FrameClassificationTask("f2", 0), "Greetings"),
        TriggerIdentificationSample(TriggerIdentificationTask("t1"), []),
    ]
    balanced_tasks = balance_tasks_by_type(tasks, max_duplication_factor=10)
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


def test_balance_tasks_by_type_caps_replications() -> None:
    tasks = [
        ArgumentsExtractionSample(ArgumentsExtractionTask("a1", 0, "Greetings"), []),
        ArgumentsExtractionSample(ArgumentsExtractionTask("a2", 0, "Greetings"), []),
        ArgumentsExtractionSample(ArgumentsExtractionTask("a3", 0, "Greetings"), []),
        ArgumentsExtractionSample(ArgumentsExtractionTask("a4", 0, "Greetings"), []),
        TriggerIdentificationSample(TriggerIdentificationTask("t1"), []),
    ]
    capped_balanced_tasks = balance_tasks_by_type(tasks, max_duplication_factor=2)
    assert len(capped_balanced_tasks) == 6  # 4 arg, 2 trigger
