from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import Adam

from torchtextlogic.models.prover import PRover


class PLPRover(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = "roberta-base",
        learning_rate: float = 1e-5,
        weight_decay=0.1,
    ) -> None:
        super().__init__()
        self.model = PRover(pretrained_model)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, proof_offsets=None, node_labels=None, edge_labels=None, qa_labels=None):  # type: ignore
        return self.model(x, proof_offsets, node_labels, edge_labels, qa_labels)

    def predict(self, triples, rules, question, device: str = "cpu"):
        return self.model.predict(triples, rules, question, device)

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        x, p_of, n_y, e_y, qa_y = train_batch
        loss = self(x, p_of, n_y, e_y, qa_y)
        self.log("train_loss", loss[0], on_epoch=True)
        return loss[0]

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        x, p_of, n_y, e_y, qa_y = val_batch
        loss = self(x, p_of, n_y, e_y, qa_y)
        self.log("vall loss", loss[0], on_epoch=True)
