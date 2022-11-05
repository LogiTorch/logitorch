from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from logitorch.models.prover import PRover


class PLPRover(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = "roberta-base",
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.model = PRover(pretrained_model, num_labels=num_labels)
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, proof_offsets=None, node_labels=None, edge_labels=None, qa_labels=None, device: str = "cpu"):  # type: ignore
        return self.model(
            x, proof_offsets, node_labels, edge_labels, qa_labels, device=device
        )

    def predict(self, triples, rules, question, device: str = "cpu"):
        return self.model.predict(triples, rules, question, device)

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
        x, p_of, n_y, e_y, qa_y = train_batch
        loss = self(x, p_of, n_y, e_y, qa_y, device=self.device)
        self.log("train_loss", loss[0], on_epoch=True)
        return loss[0]

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        x, p_of, n_y, e_y, qa_y = val_batch
        loss = self(x, p_of, n_y, e_y, qa_y, device=self.device)
        self.log("val_loss", loss[0], on_epoch=True)
