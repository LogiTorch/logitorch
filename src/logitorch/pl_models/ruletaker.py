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
        """
        Initializes the PLRuleTaker module.

        Args:
            learning_rate (float): The learning rate for the optimizer. Default is 1e-5.
            weight_decay (float): The weight decay for the optimizer. Default is 0.1.
            num_labels (int): The number of labels for the RuleTaker model. Default is 2.
        """
        super().__init__()
        self.model = RuleTaker(num_labels=num_labels)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, y):  # type: ignore
        """
        Performs a forward pass of the PLRuleTaker module.

        Args:
            x: The input data.
            y: The target labels.

        Returns:
            The output of the model.
        """
        return self.model(x, y)

    def predict(self, context: str, question: str, device: str = "cpu") -> int:
        """
        Predicts the label for a given context and question.

        Args:
            context (str): The context.
            question (str): The question.
            device (str): The device to use for prediction. Default is "cpu".

        Returns:
            The predicted label.
        """
        return self.model.predict(context, question, device)

    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler for training.

        Returns:
            The optimizer and scheduler.
        """
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
        """
        Performs a training step.

        Args:
            train_batch: The batch of training data.
            batch_idx (int): The index of the batch.

        Returns:
            The training loss.
        """
        x, y = train_batch
        loss = self(x, y)
        self.log("train_loss", loss[0], on_epoch=True)
        return loss[0]

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        """
        Performs a validation step.

        Args:
            val_batch: The batch of validation data.
            batch_idx (int): The index of the batch.
        """
        x, y = val_batch
        loss = self(x, y)
        self.log("val_loss", loss[0], on_epoch=True)
