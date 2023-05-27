from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from logitorch.models.ruletaker import RuleTaker


class PLRuleTaker(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.model = RuleTaker(num_labels=num_labels)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, y):  # type: ignore
        return self.model(x, y)

    def predict(self, context: str, question: str, device: str = "cpu") -> int:
        return self.model.predict(context, question, device)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        x, y = train_batch
        loss = self(x, y)
        self.log("train_loss", loss[0], on_epoch=True)
        return loss[0]

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        x, y = val_batch
        loss = self(x, y)
        self.log("val_loss", loss[0], on_epoch=True)
