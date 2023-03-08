from __future__ import annotations
from dataclasses import replace
from typing import Callable

from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionSample,
    FrameClassificationSample,
    TaskSample,
    TriggerIdentificationSample,
)


def is_valid_splice(
    start_loc: int,
    delete_num: int,
    critical_indices: list[int],
) -> bool:
    """
    Helper to check if a splice is valid. A splice is valid if it does not delete any of the critical indices.
    """
    for index in critical_indices:
        if index >= start_loc and index < start_loc + delete_num:
            return False
    return True


def splice_text(
    task_sample: TaskSample,
    modify_text_cb: Callable[[str, list[int]], tuple[int, int, str] | None],
) -> TaskSample:
    """
    Helper to modify the text of a TaskSample that may change the length of the text. This is
    a more complex augmentation since it requires rewriting indices, and can potentially break
    the sample. This is loosely modified on the `splice()` function from javascript, where
    a start position is given, followed by the number of chars to remove, and then the new
    string to insert.

    This takes the task sample and a lambda function. The lambda funtion that takes the text of the
    task sample and a list of critical indices to the task, which must not be deleted during the splice.
    The lambda returns a tuple of the start position, the number of chars to remove, and the new string
    to insert.
    """

    def modify_text(
        text: str, critical_indices: list[int]
    ) -> tuple[str, Callable[[int], int]]:
        modify_results = modify_text_cb(text, critical_indices)
        if modify_results is None:
            return text, lambda i: i
        start_loc, delete_num, insert_text = modify_results
        if not is_valid_splice(start_loc, delete_num, critical_indices):
            raise ValueError(
                f"Critical index was deleted during splice. This is not allowed: {text}, {start_loc}, {delete_num}"
            )
        index_modifier = (
            lambda i: i if i <= start_loc else i + len(insert_text) - delete_num
        )
        new_text = text[:start_loc] + insert_text + text[start_loc + delete_num :]
        return new_text, index_modifier

    if isinstance(task_sample, ArgumentsExtractionSample):
        critical_indices = [task_sample.task.trigger_loc]
        for frame_element in task_sample.frame_elements:
            critical_indices.append(frame_element.start_loc)
            critical_indices.append(frame_element.end_loc)
        new_text, index_modifier = modify_text(task_sample.task.text, critical_indices)
        return ArgumentsExtractionSample(
            frame_elements=[
                replace(
                    elm,
                    start_loc=index_modifier(elm.start_loc),
                    end_loc=index_modifier(elm.end_loc),
                )
                for elm in task_sample.frame_elements
            ],
            task=replace(
                task_sample.task,
                text=new_text,
                trigger_loc=index_modifier(task_sample.task.trigger_loc),
            ),
        )

    if isinstance(task_sample, FrameClassificationSample):
        critical_indices = [task_sample.task.trigger_loc]
        new_text, index_modifier = modify_text(task_sample.task.text, critical_indices)
        return FrameClassificationSample(
            frame=task_sample.frame,
            task=replace(
                task_sample.task,
                text=new_text,
                trigger_loc=index_modifier(task_sample.task.trigger_loc),
            ),
        )

    if isinstance(task_sample, TriggerIdentificationSample):
        critical_indices = task_sample.trigger_locs
        new_text, index_modifier = modify_text(task_sample.task.text, critical_indices)
        return TriggerIdentificationSample(
            trigger_locs=[index_modifier(loc) for loc in task_sample.trigger_locs],
            task=replace(task_sample.task, text=new_text),
        )

    raise ValueError(f"Unknown task sample type: {type(task_sample)}")
