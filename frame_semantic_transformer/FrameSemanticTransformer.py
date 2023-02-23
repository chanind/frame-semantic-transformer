from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, cast
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from frame_semantic_transformer.constants import (
    MODEL_MAX_LENGTH,
    MODEL_REVISION,
    OFFICIAL_RELEASES,
)
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache
from frame_semantic_transformer.data.data_utils import chunk_list, marked_string_to_locs
from frame_semantic_transformer.data.loaders.loader import InferenceLoader
from frame_semantic_transformer.data.loaders.framenet17 import Framenet17InferenceLoader
from frame_semantic_transformer.predict import batch_predict
from frame_semantic_transformer.data.tasks import (
    ArgumentsExtractionTask,
    FrameClassificationTask,
    Task,
    TriggerIdentificationTask,
)

# has the form {sentence: {trigger_loc: (frame, {frame_element: text})}}
ResultsAccumulator = Dict[str, Dict[int, Tuple[str, Dict[str, str]]]]


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
    _model: T5ForConditionalGeneration | None = None
    _tokenizer: T5TokenizerFast | None = None
    model_path: str
    model_revision: str | None = None
    device: torch.device
    batch_size: int
    predictions_per_sample: int
    loader_cache: LoaderDataCache

    def __init__(
        self,
        model_name_or_path: str = "base",
        use_gpu: bool = torch.cuda.is_available(),
        batch_size: int = 8,
        predictions_per_sample: int = 5,
        inference_loader: Optional[InferenceLoader] = None,
    ):
        self.model_path = model_name_or_path
        if model_name_or_path in OFFICIAL_RELEASES:
            self.model_path = f"chanind/frame-semantic-transformer-{model_name_or_path}"
            self.model_revision = MODEL_REVISION
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.batch_size = batch_size
        self.predictions_per_sample = predictions_per_sample
        loader = inference_loader or Framenet17InferenceLoader()
        self.loader_cache = LoaderDataCache(loader)

    def setup(self) -> None:
        """
        Initialize the model and tokenizer, and download models / files as needed
        If this is not called explicitly it will be lazily called before inference
        """
        self._model = T5ForConditionalGeneration.from_pretrained(
            self.model_path, revision=self.model_revision
        ).to(self.device)
        self._tokenizer = T5TokenizerFast.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            model_max_length=MODEL_MAX_LENGTH,
        )
        self.loader_cache.setup()
        self._validate_loader()

    def _validate_loader(self) -> None:
        """
        Helper to ensure that the loader being used matches the one used to train the model
        otherwise results will be potentially nonsensical
        """
        loader = self.loader_cache.loader
        config = self.model.config
        expected_loader_name = (
            config.inference_loader
            if hasattr(config, "inference_loader")
            else loader.name()
        )
        if expected_loader_name != loader.name():
            raise ValueError(
                f"Model {self.model_path} was trained with inference loader {expected_loader_name} "
                f"but you are using {loader.name()}. Please pass in the correct loader as 'inference_loader' "
                f"when initializing FrameSemanticTransfomer."
            )

    @property
    def model(self) -> T5ForConditionalGeneration:
        if not self._model:
            self.setup()
        return cast(T5ForConditionalGeneration, self._model)

    @property
    def tokenizer(self) -> T5TokenizerFast:
        if not self._tokenizer:
            self.setup()
        return cast(T5TokenizerFast, self._tokenizer)

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

    def detect_frames(self, sentence: str) -> DetectFramesResult:
        return self.detect_frames_bulk([sentence])[0]

    def _collate_results(
        self, results_acc: ResultsAccumulator
    ) -> dict[str, DetectFramesResult]:
        """
        helper to turn the results accumulator into DetectFramesResults
        """
        results: dict[str, DetectFramesResult] = {}
        for sentence, frame_data in results_acc.items():
            frame_results: list[FrameResult] = []
            for trigger_loc, (frame, elements) in frame_data.items():
                frame_results.append(
                    FrameResult(
                        name=frame,
                        trigger_location=trigger_loc,
                        frame_elements=[
                            FrameElementResult(element, text)
                            for element, text in elements.items()
                        ],
                    )
                )
            results[sentence] = DetectFramesResult(
                sentence,
                trigger_locations=list(frame_data.keys()),
                frames=frame_results,
            )
        return results

    def detect_frames_bulk(self, sentences: Iterable[str]) -> list[DetectFramesResult]:
        tasks_queue: list[Task] = []
        # slowly build up results from each task as they complete
        results_acc: ResultsAccumulator = defaultdict(dict)
        # T5 doesn't necessarily have to output the original sentence, even though it's supposed to
        # This map just keeps track of the original sentence for each output sentence so we can match them up
        parsed_sentences_map: dict[str, str] = {}
        for sentence in sentences:
            tasks_queue.append(TriggerIdentificationTask(text=sentence))
        while len(tasks_queue) > 0:
            batch = tasks_queue[: self.batch_size]
            tasks_queue = tasks_queue[self.batch_size :]
            batch_results = self._batch_predict([task.get_input() for task in batch])
            for task, preds in zip(
                batch, chunk_list(batch_results, self.predictions_per_sample)
            ):
                if isinstance(task, TriggerIdentificationTask):
                    # first identify triggers
                    result = task.parse_output(preds, self.loader_cache)
                    parsed_sent, trigger_locs = marked_string_to_locs(result)
                    parsed_sentences_map[task.text] = parsed_sent
                    for trigger_loc in trigger_locs:
                        tasks_queue.append(
                            FrameClassificationTask(
                                text=task.text,
                                trigger_loc=trigger_loc,
                                loader_cache=self.loader_cache,
                            )
                        )
                elif isinstance(task, FrameClassificationTask):
                    # then identify frames
                    frame = task.parse_output(preds, self.loader_cache)
                    if frame:
                        results_acc[task.text][task.trigger_loc] = (frame, {})
                        tasks_queue.append(
                            ArgumentsExtractionTask(
                                text=task.text,
                                trigger_loc=task.trigger_loc,
                                frame=frame,
                                loader_cache=self.loader_cache,
                            )
                        )
                elif isinstance(task, ArgumentsExtractionTask):
                    # finally identify frame elements
                    frame_element_tuples = task.parse_output(preds, self.loader_cache)
                    results_acc[task.text][task.trigger_loc][1].update(
                        frame_element_tuples
                    )
        results_map = self._collate_results(results_acc)
        return [results_map[parsed_sentences_map[sentence]] for sentence in sentences]
