from __future__ import annotations
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from frame_semantic_transformer.data.loaders.loader import FrameAnnotatedSentence
from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionSample,
    ArgumentsExtractionTask,
    TaskSample,
    TriggerIdentificationSample,
    TriggerIdentificationTask,
    FrameClassificationSample,
    FrameClassificationTask,
)


def tasks_from_annotated_sentences(
    annotated_sentences: list[FrameAnnotatedSentence],
    loader_cache: LoaderDataCache,
) -> list[TaskSample]:
    task_samples: list[TaskSample] = []
    for annotated_sentence in annotated_sentences:
        trigger_locs = []
        for annotation in annotated_sentence.annotations:
            for trigger_loc in annotation.trigger_locs:
                trigger_locs.append(trigger_loc)
                task_samples.append(
                    FrameClassificationSample(
                        task=FrameClassificationTask(
                            text=annotated_sentence.text,
                            trigger_loc=trigger_loc,
                            loader_cache=loader_cache,
                        ),
                        frame=annotation.frame,
                    )
                )
                task_samples.append(
                    ArgumentsExtractionSample(
                        task=ArgumentsExtractionTask(
                            text=annotated_sentence.text,
                            trigger_loc=trigger_loc,
                            frame=annotation.frame,
                            loader_cache=loader_cache,
                        ),
                        frame_elements=annotation.frame_elements,
                    )
                )

        task_samples.append(
            TriggerIdentificationSample(
                task=TriggerIdentificationTask(text=annotated_sentence.text),
                trigger_locs=trigger_locs,
            )
        )
    return task_samples
