from __future__ import annotations
from dataclasses import replace
from typing import Callable

from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionSample,
    FrameClassificationSample,
    TaskSample,
    TriggerIdentificationSample,
)


def modify_text_without_changing_length(
    task_sample: TaskSample, modify_text_cb: Callable[[str], str]
) -> TaskSample:
    """
    Helper to modify the text of a TaskSample without changing the length of the text
    This is a simple augmentation since it doesn't require rewriting indices

    This takes the task sample and a lambda function that takes the text of the task sample
    and returns the modified text. It then modifies the text of the task sample and returns
    """

    def modify_text(text: str) -> str:
        new_text = modify_text_cb(text)
        if len(new_text) != len(text):
            raise ValueError(
                f"Text length changed during augmentation: {text} -> {new_text}"
            )
        return new_text

    if isinstance(task_sample, ArgumentsExtractionSample):
        new_text = modify_text(task_sample.task.text)
        return replace(
            task_sample,
            task=replace(task_sample.task, text=new_text),
        )

    if isinstance(task_sample, FrameClassificationSample):
        new_text = modify_text(task_sample.task.text)
        return replace(
            task_sample,
            task=replace(task_sample.task, text=new_text),
        )

    if isinstance(task_sample, TriggerIdentificationSample):
        new_text = modify_text(task_sample.task.text)
        return replace(
            task_sample,
            task=replace(task_sample.task, text=new_text),
        )

    raise ValueError(f"Unknown task sample type: {type(task_sample)}")
