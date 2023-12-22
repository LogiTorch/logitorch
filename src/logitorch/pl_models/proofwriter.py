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
        """
        Initializes a PLProofWriter object.

        Args:
            pretrained_model (str, optional): The name or path of the pretrained model to use. Defaults to "google/t5-v1_1-large".
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to None.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.1.
        """
        super().__init__()
        self.model = ProofWriter(pretrained_model)
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x, y) -> SequenceClassifierOutput:  # type: ignore
        """
        Performs a forward pass of the model.

        Args:
            x: The input data.
            y: The target data.

        Returns:
            SequenceClassifierOutput: The output of the model.
        """
        return self.model(x, y)

    def predict(
        self,
        context: str,
        question: str,
        num_beams: int = 5,
        max_length: int = 120,
        device: str = "cpu",
    ):
        """
        Generates predictions for the given context and question.

        Args:
            context (str): The context for the prediction.
            question (str): The question for the prediction.
            num_beams (int, optional): The number of beams for beam search decoding. Defaults to 5.
            max_length (int, optional): The maximum length of the generated sequence. Defaults to 120.
            device (str, optional): The device to use for prediction. Defaults to "cpu".

        Returns:
            The generated predictions.
        """
        return self.model.predict(context, question, num_beams, max_length, device)

    def configure_optimizers(self):
        """
        Configures the optimizer and scheduler for training.

        Returns:
            Tuple[List[Optimizer], List[Dict[str, Any]]]: The optimizer and scheduler.
        """
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
        """
        Performs a training step.

        Args:
            train_batch (Tuple[Dict[str, torch.Tensor], torch.Tensor]): The batch of training data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        x, y = train_batch
        loss = self(x, y).loss
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
        """
        Performs a validation step.

        Args:
            val_batch (Tuple[Dict[str, torch.Tensor], torch.Tensor]): The batch of validation data.
            batch_idx (int): The index of the batch.
        """
        x, y = val_batch
        loss = self(x, y).loss
        self.log("val_loss", loss, on_epoch=True)
