from __future__ import annotations

from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionSample,
    FrameClassificationSample,
    TaskSample,
    TriggerIdentificationSample,
)


def get_sample_text(sample: TaskSample) -> str:
    if isinstance(sample, ArgumentsExtractionSample):
        return sample.task.text
    if isinstance(sample, FrameClassificationSample):
        return sample.task.text
    if isinstance(sample, TriggerIdentificationSample):
        return sample.task.text
    raise ValueError(f"Unknown sample type: {type(sample)}")
