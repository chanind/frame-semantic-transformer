from __future__ import annotations
from typing import Any, Callable, Optional
import pytorch_lightning as pl
from transformers import (
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import DataLoader, Dataset


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

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any],
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
        self.batch_size = batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:  # type: ignore
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:  # type: ignore
        loss = self._step(batch)
        return {"val_loss": loss}

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
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
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