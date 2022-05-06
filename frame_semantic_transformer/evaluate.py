from __future__ import annotations
from typing import Sequence
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.corpus import framenet as fn

from frame_semantic_transformer.data.SampleSentence import SampleSentence
from frame_semantic_transformer.data.chunk_list import chunk_list
from frame_semantic_transformer.predict import batch_predict


all_valid_frames = {frame.name for frame in fn.frames()}


def evaluate(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    samples: Sequence[SampleSentence],
    batch_size: int = 10,
) -> dict[str, list[int]]:
    results: dict[str, list[int]] = {"frame": [0, 0, 0], "args": [0, 0, 0]}
    for samples_chunk in tqdm(chunk_list(samples, batch_size)):
        frame_inputs: list[str] = []
        args_inputs: list[str] = []
        expected_frame_outputs: list[str] = []
        expected_args_outputs: list[str] = []
        for sample in samples_chunk:
            frame_inputs.append(sample.frame_classification_input)
            args_inputs.append(sample.frame_args_input)
            expected_frame_outputs.append(sample.frame)
            expected_args_outputs.append(sample.frame_elements_str)

        frame_predictions = batch_predict(model, tokenizer, frame_inputs)
        args_predictions = batch_predict(model, tokenizer, args_inputs)
        for frame_prediction, expected_output in zip(
            frame_predictions, expected_frame_outputs
        ):

            if frame_prediction == expected_output:
                results["frame"][0] += 1
            elif frame_prediction in all_valid_frames:
                results["frame"][1] += 1
            else:
                results["frame"][2] += 1

        for args_prediction, expected_output in zip(
            args_predictions, expected_args_outputs
        ):
            if args_prediction == expected_output:
                results["args"][0] += 1
            # TODO: figure out fp/fn for frame elements
            else:
                results["args"][1] += 1
    return results
