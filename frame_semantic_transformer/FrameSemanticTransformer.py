from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from frame_semantic_transformer.data.data_utils import chunk_list, marked_string_to_locs
from frame_semantic_transformer.data.tasks.ArgumentsExtractionTask import (
    ArgumentsExtractionTask,
)
from frame_semantic_transformer.data.tasks.FrameClassificationTask import (
    FrameClassificationTask,
)

from frame_semantic_transformer.data.tasks.TriggerIdentificationTask import (
    TriggerIdentificationTask,
)
from frame_semantic_transformer.predict import batch_predict


OFFICIAL_RELEASES = ["base", "small", "large"]


@dataclass
class FrameElementResult:
    name: str
    text: str


@dataclass
class FrameResult:
    name: str
    trigger_location: int
    frame_elements: list[FrameElementResult]


@dataclass
class DetectFramesResult:
    sentence: str
    trigger_locations: list[int]
    frames: list[FrameResult]


class FrameSemanticTransformer:

    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer
    device: torch.device
    max_batch_size: int
    predictions_per_sample: int

    def __init__(
        self,
        model_name_or_path: str = "base",
        use_gpu: bool = torch.cuda.is_available(),
        max_batch_size: int = 8,
        predictions_per_sample: int = 5,
    ):
        model_path = model_name_or_path
        if model_name_or_path in OFFICIAL_RELEASES:
            model_path = f"chanind/frame-semantic-transformer-{model_name_or_path}"
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.max_batch_size = max_batch_size
        self.predictions_per_sample = predictions_per_sample

    def _batch_predict(self, inputs: list[str]) -> list[str]:
        """
        helper to avoid needing to repeatedly pass in the same params every call to predict
        """
        return batch_predict(
            self.model,
            self.tokenizer,
            inputs,
            num_beams=self.predictions_per_sample,
            num_return_sequences=self.predictions_per_sample,
        )

    def _identify_triggers(self, sentence: str) -> tuple[str, list[int]]:
        task = TriggerIdentificationTask(text=sentence)
        outputs = self._batch_predict([task.get_input()])
        result = task.parse_output(outputs)
        return marked_string_to_locs(result)

    def _classify_frames(
        self, sentence: str, trigger_locs: list[int]
    ) -> list[str | None]:
        """
        Return a list containing a frame for each trigger_loc passed in.
        If no frame can be found, None is returned for the frame instead.
        """
        frame_classification_tasks: list[FrameClassificationTask] = []
        frames: list[str | None] = []

        for trigger_loc in trigger_locs:
            frame_classification_tasks.append(
                FrameClassificationTask(text=sentence, trigger_loc=trigger_loc)
            )
        for batch in chunk_list(
            frame_classification_tasks, chunk_size=self.max_batch_size
        ):
            batch_results = self._batch_predict([task.get_input() for task in batch])
            for preds, frame_task in zip(
                chunk_list(batch_results, self.predictions_per_sample),
                batch,
            ):
                frames.append(frame_task.parse_output(preds))
        return frames

    def _extract_frame_args(
        self, sentence: str, frame_with_trigger_locs: list[tuple[str, int]]
    ) -> list[list[tuple[str, str]]]:
        """
        return a list of tuples of (frame_element, text) for each frame/trigger loc passed in.
        The returned list will have the same length as the frame_with_trigger_locs list,
        with each element corresponding to a frame/loc in the input list
        """
        frame_element_results: list[list[tuple[str, str]]] = []
        arg_extraction_tasks = [
            ArgumentsExtractionTask(
                text=sentence,
                trigger_loc=trigger_loc,
                frame=frame,
            )
            for frame, trigger_loc in frame_with_trigger_locs
        ]
        for args_tasks_batch in chunk_list(
            arg_extraction_tasks, chunk_size=self.max_batch_size
        ):
            batch_results = self._batch_predict(
                [task.get_input() for task in args_tasks_batch],
            )
            for preds, args_task in zip(
                chunk_list(batch_results, self.predictions_per_sample), args_tasks_batch
            ):
                frame_element_results.append(args_task.parse_output(preds))
        return frame_element_results

    def detect_frames(self, sentence: str) -> DetectFramesResult:
        # first detect trigger locations
        base_sentence, trigger_locs = self._identify_triggers(sentence)
        # next detect frames for each trigger
        frames = self._classify_frames(base_sentence, trigger_locs)

        frame_and_locs = [
            (frame, loc) for frame, loc in zip(frames, trigger_locs) if frame
        ]
        frame_elements_lists = self._extract_frame_args(base_sentence, frame_and_locs)
        frame_results: list[FrameResult] = []
        for ((frame, loc), frame_element_tuples) in zip(
            frame_and_locs, frame_elements_lists
        ):
            frame_elements = [
                FrameElementResult(element, text)
                for element, text in frame_element_tuples
            ]
            frame_results.append(
                FrameResult(
                    name=frame,
                    trigger_location=loc,
                    frame_elements=frame_elements,
                )
            )
        return DetectFramesResult(
            base_sentence,
            trigger_locations=trigger_locs,
            frames=frame_results,
        )
