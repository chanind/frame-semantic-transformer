from __future__ import annotations
from transformers import T5Tokenizer
from nltk.corpus import framenet as fn
from frame_semantic_transformer.data.SentenceDataset import SentenceDataset

from frame_semantic_transformer.data.load_framenet_samples import (
    parse_samples_from_lexical_unit,
)


def test_SentenceDataset() -> None:
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    lu = fn.lu(6403)  # repair.v
    # use the first 4 samples
    samples = parse_samples_from_lexical_unit(lu)[0:4]

    dataset = SentenceDataset(samples, tokenizer)

    assert len(dataset) == 8
    assert len(dataset[0]["input_ids"]) == 64
    assert len(dataset[0]["attention_mask"]) == 64
    assert len(dataset[0]["labels"]) == 46
