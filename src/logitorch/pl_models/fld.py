from typing import Dict, Tuple, Optional

import pytorch_lightning as pl
import torch
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
from logitorch.models.fld import FLDAllAtOnceProver


class PLFLDAllAtOnceProver(pl.LightningModule):
    """
    PyTorch Lightning module for Fine-tuned Language Decoder (FLD) all-at-once prover.

    Args:
        pretrained_model (str): Pretrained model name or path (default: "t5-base").
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer (default: 0.1).
        warmup_steps (int, optional): Number of warmup steps for learning rate scheduler (default: 1000).

    Attributes:
        model (FLDAllAtOnceProver): FLD model.
        pretrained_model (str): Pretrained model name or path.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        warmup_steps (int): Number of warmup steps for learning rate scheduler.
        optimizer (AdamW): Optimizer for training.
    """

    def __init__(
        self,
        pretrained_model: str = "t5-base",
        learning_rate: float = None,
        weight_decay=0.1,
        warmup_steps: Optional[int] = 1000,
    ) -> None:
        super().__init__()
        self.model = FLDAllAtOnceProver(pretrained_model)
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.optimizer = None

    def forward(self, x, y) -> SequenceClassifierOutput:
        """
        Forward pass of the model.

        Args:
            x: Input data.
            y: Target data.

        Returns:
            SequenceClassifierOutput: Model output.
        """
        return self.model(x, y)

    def predict(
        self,
        prompt: str,
        num_beams: int = 5,
        max_length: int = 1000,
        device: str = "cpu",
    ):
        """
        Generate predictions using the model.

        Args:
            prompt (str): Input prompt.
            num_beams (int): Number of beams for beam search (default: 5).
            max_length (int): Maximum length of generated sequence (default: 1000).
            device (str): Device to use for prediction (default: "cpu").

        Returns:
            Model predictions.
        """
        return self.model.predict(prompt, num_beams, max_length, device)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[List[Optimizer], List[Dict[str, Any]]]: Optimizers and schedulers.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        warmup_steps = self.warmup_steps or int(0.1 * self.trainer.estimated_stepping_batches)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        self.optimizer = optimizer
        return [optimizer], [scheduler]

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            train_batch: Batch of training data.
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = train_batch
        loss = self(x, y).loss
        self.log("train_loss", loss, on_step=True)

        for param_group in self.optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
        return loss

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:
        """
        Validation step.

        Args:
            val_batch: Batch of validation data.
            batch_idx: Index of the batch.
        """
        x, y = val_batch
        loss = self(x, y).loss
        self.log("val_loss", loss, on_epoch=True)
