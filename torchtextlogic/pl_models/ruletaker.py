from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers.modeling_outputs import SequenceClassifierOutput

from torchtextlogic.models.ruletaker import RuleTaker


class PLRuleTaker(pl.LightningModule):
    def __init__(self, pretrained_model: str, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.model = RuleTaker(pretrained_model)
        self.learning_rate = learning_rate

    def forward(self, x, y):  # type: ignore
        return self.model(x, y)

    def predict(self, context: str, question: str, device: str = "cpu"):
        return self.model.predict(context, question, device)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        x, y = train_batch
        loss = self(x, y).loss
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        x, y = val_batch
        loss = self(x, y).loss
        self.log("val_loss", loss, on_epoch=True)
