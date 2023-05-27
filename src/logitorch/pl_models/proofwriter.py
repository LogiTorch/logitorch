from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from transformers import Adafactor, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from logitorch.models.proofwriter import ProofWriter


class PLProofWriter(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = "google/t5-v1_1-large",
        learning_rate: float = None,
        weight_decay=0.1,
    ) -> None:
        super().__init__()
        self.model = ProofWriter(pretrained_model)
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, y) -> SequenceClassifierOutput:  # type: ignore
        return self.model(x, y)

    def predict(
        self,
        context: str,
        question: str,
        num_beams: int = 5,
        max_length: int = 120,
        device: str = "cpu",
    ):
        return self.model.predict(context, question, num_beams, max_length, device)

    def configure_optimizers(self):
        if self.learning_rate is None:
            optimizer = Adafactor(
                self.model.parameters(),
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            return optimizer
        else:
            optimizer = Adafactor(
                self.model.parameters(),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
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
        loss = self(x, y).loss
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        x, y = val_batch
        loss = self(x, y).loss
        self.log("val_loss", loss, on_epoch=True)
