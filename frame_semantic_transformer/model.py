from __future__ import annotations
from typing import Any, Optional
import pytorch_lightning as pl
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import DataLoader, Dataset

from frame_semantic_transformer.data.SentenceDataset import SentenceDataset
from frame_semantic_transformer.data.load_framenet_samples import load_framenet_samples
from frame_semantic_transformer.data.shuffle_and_split import shuffle_and_split


class T5FineTuner(pl.LightningModule):
    """
    Based on https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
    """

    lr: float
    eps: float
    weight_decay: float
    batch_size: int
    train_dataset: Dataset[Any]
    val_dataset: Dataset[Any]
    n_gpu: int
    num_train_epochs: int
    warmup_steps: int
    gradient_accumulation_steps: int
    model: T5ForConditionalGeneration
    tokenizer: T5Tokenizer

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        lr: float = 1e-4,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        batch_size: int = 8,
        n_gpu: int = 0,
        num_train_epochs: int = 2,
        gradient_accumulation_steps: int = 16,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps

        samples = load_framenet_samples()
        train_samples, val_samples = shuffle_and_split(samples, 0.8)
        self.train_dataset = SentenceDataset(train_samples, tokenizer)
        self.val_dataset = SentenceDataset(val_samples, tokenizer)

    def is_logger(self) -> bool:
        # keep mypy happy...
        if not self.trainer:
            return False
        return self.trainer.global_rank <= 0

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.model(*args, **kwargs)

    def _step(self, batch: Any) -> Any:
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["labels"],
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:  # type: ignore
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs: Any) -> dict[str, Any]:  # type: ignore
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {
            "avg_train_loss": avg_train_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:  # type: ignore
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs) -> dict[str, Any]:  # type: ignore
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {
            "avg_val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def configure_optimizers(self) -> list[AdamW]:
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters: Any = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=self.eps,
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(  # type: ignore
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Any,
        optimizer_idx: int,
        second_order_closure: Optional[Any] = None,
    ) -> None:
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self) -> dict[str, Any]:
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss),  # type: ignore
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }

        return tqdm_dict

    def train_dataloader(self) -> DataLoader[Any]:
        dataloader: Any = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
        )
        t_total = (
            (len(dataloader.dataset) // (self.batch_size * max(1, self.n_gpu)))
            // self.gradient_accumulation_steps
            * float(self.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)
