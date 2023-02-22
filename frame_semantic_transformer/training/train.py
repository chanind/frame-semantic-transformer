from __future__ import annotations
import logging
from typing import Literal, Optional, Union
import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from frame_semantic_transformer.constants import DEFAULT_NUM_WORKERS, MODEL_MAX_LENGTH
from frame_semantic_transformer.data.LoaderDataCache import LoaderDataCache

from frame_semantic_transformer.data.TaskSampleDataset import TaskSampleDataset
from frame_semantic_transformer.data.loaders.framenet17 import (
    Framenet17InferenceLoader,
    Framenet17TrainingLoader,
)
from frame_semantic_transformer.data.loaders.loader import (
    InferenceLoader,
    TrainingLoader,
)
from frame_semantic_transformer.data.tasks_from_annotated_sentences import (
    tasks_from_annotated_sentences,
)
from frame_semantic_transformer.training.TrainingDataModule import TrainDataModule
from frame_semantic_transformer.training.TrainingModelWrapper import (
    TrainingModelWrapper,
)

logger = logging.getLogger(__name__)


def train(
    base_model: str = "t5-base",
    batch_size: int = 8,
    max_epochs: int = 10,
    use_gpu: bool = torch.cuda.is_available(),
    output_dir: str = "outputs",
    early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
    precision: Union[Literal[64, 32, 16], Literal["64", "32", "16", "bf16"]] = 32,
    lr: float = 1e-4,
    num_workers: int = DEFAULT_NUM_WORKERS,
    save_only_last_epoch: bool = False,
    balance_tasks: bool = True,
    max_task_duplication_factor: int = 2,
    skip_initial_epochs_validation: int = 0,
    inference_loader: Optional[InferenceLoader] = None,
    training_loader: Optional[TrainingLoader] = None,
) -> tuple[T5ForConditionalGeneration, T5TokenizerFast]:
    device = torch.device("cuda" if use_gpu else "cpu")
    logger.info("loading base T5 model")
    model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(
        base_model, model_max_length=MODEL_MAX_LENGTH
    )
    if not inference_loader:
        inference_loader = Framenet17InferenceLoader()
    loader_cache = LoaderDataCache(inference_loader)
    if not training_loader:
        training_loader = Framenet17TrainingLoader()

    model.config.training_loader = training_loader.name()
    model.config.inference_loader = inference_loader.name()

    logger.info("loading train/test/val datasets")
    training_data = training_loader.load_training_data()
    validation_data = training_loader.load_validation_data()
    test_data = training_loader.load_test_data()

    train_dataset = TaskSampleDataset(
        tasks_from_annotated_sentences(training_data, loader_cache),
        tokenizer,
        balance_tasks=balance_tasks,
        max_task_duplication_factor=max_task_duplication_factor,
        augmentations=training_loader.get_augmentations(),
    )
    val_dataset = TaskSampleDataset(
        tasks_from_annotated_sentences(validation_data, loader_cache),
        tokenizer,
        balance_tasks=False,
    )
    test_dataset = TaskSampleDataset(
        tasks_from_annotated_sentences(test_data, loader_cache),
        tokenizer,
        balance_tasks=False,
    )

    data_module = TrainDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model_wrapper = TrainingModelWrapper(
        model,
        tokenizer,
        lr=lr,
        output_dir=output_dir,
        save_only_last_epoch=save_only_last_epoch,
        skip_initial_epochs_validation=skip_initial_epochs_validation,
        loader_cache=loader_cache,
    )

    # add callbacks
    callbacks: list[Callback] = [TQDMProgressBar(refresh_rate=5)]

    if early_stopping_patience_epochs > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=early_stopping_patience_epochs,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    # prepare trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=1 if use_gpu else 0,
        precision=precision,
        log_every_n_steps=1,
    )

    logger.info("beginning training")

    # fit trainer
    trainer.fit(model_wrapper, data_module)

    return model, tokenizer
