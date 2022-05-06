from __future__ import annotations
from typing import Iterable
from transformers import T5Tokenizer, T5ForConditionalGeneration


def predict(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    source_text: str,
    max_length: int = 512,
    num_return_sequences: int = 1,
    num_beams: int = 5,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 2.5,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> list[str]:
    input_ids = tokenizer.encode(
        source_text, return_tensors="pt", add_special_tokens=True
    )
    input_ids = input_ids.to(model.device)
    generated_ids = model.generate(
        input_ids=input_ids,
        num_beams=num_beams,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
    )
    preds = [
        tokenizer.decode(
            g,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        for g in generated_ids
    ]
    return preds


def batch_predict(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    source_texts: Iterable[str],
    max_length: int = 512,
    num_return_sequences: int = 1,
    num_beams: int = 5,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 2.5,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> list[str]:
    input_encoding = tokenizer(
        source_texts,
        padding="longest",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    generated_ids = model.generate(
        input_ids=input_encoding.input_ids,
        attention_mask=input_encoding.attention_mask,
        num_beams=num_beams,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
    )
    preds = [
        tokenizer.decode(
            g,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        for g in generated_ids
    ]
    return preds
