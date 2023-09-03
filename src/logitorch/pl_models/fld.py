from typing import Dict, Tuple, Optional, Union
import math

import pytorch_lightning as pl
import torch
from transformers import Adafactor, get_linear_schedule_with_warmup, AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
from logitorch.models.fld import FLDSimpleProver


class PLFLDSimpleProver(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = "google/t5-v1_1-large",
        learning_rate: float = None,
        weight_decay=0.1,
        warmup_steps: Optional[int] = 1000,
    ) -> None:
        super().__init__()
        self.model = FLDSimpleProver(pretrained_model)
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.optimizer = None

    def forward(self, x, y) -> SequenceClassifierOutput:  # type: ignore
        return self.model(x, y)

    def predict(
        self,
        prompt: str,
        num_beams: int = 5,
        max_length: int = 1000,
        device: str = "cpu",
    ):
        return self.model.predict(prompt, num_beams, max_length, device)

    def configure_optimizers(self):

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        warmup_steps = self.warmup_steps or int(0.1 * self.estimated_stepping_batches)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        self.optimizer = optimizer
        return [optimizer], [scheduler]

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        x, y = train_batch
        loss = self(x, y).loss
        self.log("train_loss", loss, on_step=True)

        for param_group in self.optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
        return loss

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        x, y = val_batch
        loss = self(x, y).loss
        self.log("val_loss", loss, on_epoch=True)


    @property
    def estimated_stepping_batches(self) -> Union[int, float]:
        """re-implementation of trainer.estimated_stepping_batches as it seems not to correctly calculate the results"""
        trainer = self.trainer
        accumulation_scheduler = trainer.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise ValueError(
                "Estimated stepping batches cannot be computed with different"
                " `accumulate_grad_batches` at different epochs."
            )

        # infinite training
        if trainer.max_epochs == -1 and trainer.max_steps == -1:
            return float("inf")

        if trainer.train_dataloader is None:
            # rank_zero_info("Loading `train_dataloader` to estimate number of stepping batches.")
            trainer.reset_train_dataloader()

        total_batches = trainer.num_training_batches

        # iterable dataset
        if total_batches == float("inf"):
            return trainer.max_steps

        trainer.accumulate_grad_batches = accumulation_scheduler.get_accumulate_grad_batches(trainer.current_epoch)
        effective_batch_size = trainer.accumulate_grad_batches
        max_estimated_steps = math.ceil(total_batches / effective_batch_size) * max(trainer.max_epochs, 1)

        max_estimated_steps = max(max_estimated_steps, trainer.max_steps) if trainer.max_steps != -1 else max_estimated_steps
        return max_estimated_steps
