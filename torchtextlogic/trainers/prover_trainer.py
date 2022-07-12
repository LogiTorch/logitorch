from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers.modeling_outputs import SequenceClassifierOutput

from torchtextlogic.models.proofwriter import ProofWriter
from torchtextlogic.models.prover import PRover


class PRoverTrainer(pl.LightningModule):
    """[summary]"""

    def __init__(
        self, pretrained_model: str = "t5-large", learning_rate: float = 1e-3
    ) -> None:
        """[summary]

        :param pretrained_model: [description]
        :type pretrained_model: str
        :param learning_rate: [description], defaults to 1e-3
        :type learning_rate: float, optional
        """
        super().__init__()
        self.model = PRover(pretrained_model)
        self.learning_rate = learning_rate

    def forward(self, x, proof_offsets=None, node_labels=None, edge_labels=None, qa_labels=None):  # type: ignore
        """[summary]

        :return: [description]
        :rtype: SequenceClassifierOutput
        """
        return self.model(x, proof_offsets, node_labels, edge_labels, qa_labels)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        """[summary]

        :param train_batch: [description]
        :type train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor]
        :param batch_idx: [description]
        :type batch_idx: int
        :return: [description]
        :rtype: torch.Tensor
        """
        x, p_of, n_y, e_y, qa_y = train_batch
        loss = self(x, p_of, n_y, e_y, qa_y)
        self.log("train_loss", loss[0], on_epoch=True)
        return loss[0]

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        """[summary]

        :param val_batch: [description]
        :type val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor]
        :param batch_idx: [description]
        :type batch_idx: int
        """
        x, p_of, n_y, e_y, qa_y = val_batch
        loss = self(x, p_of, n_y, e_y, qa_y)
        self.log("vall loss", loss[0], on_epoch=True)

    def test_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        """[summary]

        :param val_batch: [description]
        :type val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor]
        :param batch_idx: [description]
        :type batch_idx: int
        """
        x, p_of, n_y, e_y, qa_y = val_batch
        loss = self(x, p_of, n_y, e_y, qa_y)
        self.log("vall loss", loss[0], on_epoch=True)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: Optional[int] = 0) -> torch.Tensor:  # type: ignore
        """[summary]

        :param batch: [description]
        :type batch: Dict[str, torch.Tensor]
        :param batch_idx: [description]
        :type batch_idx: int
        :param dataloader_idx: [description], defaults to 0
        :type dataloader_idx: Optional[int], optional
        :return: [description]
        :rtype: torch.Tensor
        """
        outputs = self(batch)
        return outputs.logits
