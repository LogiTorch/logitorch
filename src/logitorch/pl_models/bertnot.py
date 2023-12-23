from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from logitorch.data_collators.bertnot_collator import BERTNOTWiki20KCollator
from logitorch.datasets.mlm.wiki20k_dataset import Wiki20KDataset
from logitorch.models.bertnot import BERTNOT


class PLBERTNOT(pl.LightningModule):
    """
    PyTorch Lightning module for BERTNOT model.

    Args:
        pretrained_model (str): Pretrained model name or path.
        task (str): Task type, either "mlm" (masked language modeling) or "te" (text entailment).
        num_labels (int): Number of labels for the classification task.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        batch_size (int): Batch size for the data loader.
        gamma (float): Gamma value for the loss calculation.

    Attributes:
        model (BERTNOT): BERTNOT model instance.
        pretrained_model (str): Pretrained model name or path.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        batch_size (int): Batch size for the data loader.
        gamma (float): Gamma value for the loss calculation.
        task (str): Task type, either "mlm" (masked language modeling) or "te" (text entailment).
    """

    def __init__(
        self,
        pretrained_model: str,
        task: str = "mlm",
        num_labels: int = 2,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.1,
        batch_size: int = 32,
        gamma: float = 0.4,
    ) -> None:
        super().__init__()
        self.model = BERTNOT(pretrained_model, num_labels=num_labels)
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.task = task

        if self.task == "mlm":
            self.automatic_optimization = False

    def forward(self, x, y=None, loss="cross_entropy"):
        """
        Forward pass of the PLBERTNOT model.

        Args:
            x: Input data.
            y: Target labels.
            loss (str): Loss function type.

        Returns:
            torch.Tensor: Model output.
        """
        if self.task == "mlm":
            if y is not None:
                return self.model(x, y, task="mlm", loss=loss)
            return self.model(x, task="mlm")
        else:
            if y is not None:
                return self.model(x, y, task="te")
            return self.model(x, task="te")

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: Optimizers and schedulers.
        """
        paramaters = [
            {"params": self.model.model.parameters()},
            {"params": self.model.sequence_classifier.parameters()},
        ]
        optimizer = AdamW(
            paramaters, lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def train_dataloader(self):
        """
        Get the training data loader.

        Returns:
            Dict[str, DataLoader]: Dictionary of data loaders.
        """
        negated_wiki20k_dataset = Wiki20KDataset("negated_lm_wiki20k")
        positive_wiki20k_dataset = Wiki20KDataset("positive_lm_wiki20k")
        wiki20k_dataset = Wiki20KDataset("lm_wiki20k")

        collator_fn = BERTNOTWiki20KCollator(self.pretrained_model)

        loader_negated_wiki20k = DataLoader(
            negated_wiki20k_dataset, self.batch_size, collate_fn=collator_fn
        )
        loader_positive_wiki20k = DataLoader(
            positive_wiki20k_dataset, self.batch_size, collate_fn=collator_fn
        )
        loader_wiki20k = DataLoader(
            wiki20k_dataset, self.batch_size, collate_fn=collator_fn
        )

        return {
            "negated_wiki20k": loader_negated_wiki20k,
            "positive_wiki20k": loader_positive_wiki20k,
            "wiki20k": loader_wiki20k,
        }

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int):
        """
        Training step of the PLBERTNOT model.

        Args:
            train_batch: Batch of training data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        if self.task == "mlm":
            optimizer = self.optimizers()
            x, y = train_batch["negated_wiki20k"]
            loss_ul_negated_wiki20k, _ = self(x, y, loss="unlikelihood")
            x, y = train_batch["positive_wiki20k"]
            loss_kl_positive_wiki20k, _ = self(x, y, loss="kl")

            loss = (
                self.gamma * loss_ul_negated_wiki20k
                + (1 - self.gamma) * loss_kl_positive_wiki20k
            )
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

            x, y = train_batch["wiki20k"]
            loss_kl_wiki20k, _ = self(x, y, loss="kl")
            optimizer.zero_grad()
            self.manual_backward(loss_kl_wiki20k)
            optimizer.step()

            self.log_dict(
                {
                    "unlikelihood loss negated wiki20k": loss_ul_negated_wiki20k,
                    "knowledge distillation loss positive wiki20k": loss_kl_positive_wiki20k,
                    "knowledge distillation loss wiki20k": loss_kl_wiki20k,
                },
                prog_bar=True,
            )
        else:
            x, y = train_batch
            loss, _ = self(x, y)
            self.log_dict({"train_loss": loss}, prog_bar=True, on_epoch=True)
            return loss

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int):
        """
        Validation step of the PLBERTNOT model.

        Args:
            val_batch: Batch of validation data.
            batch_idx (int): Batch index.
        """
        if self.task == "mlm":
            x, y = val_batch["negated_wiki20k"]
            loss_ul_negated_wiki20k, _ = self(x, y, loss="unlikelihood")
            x, y = val_batch["positive_wiki20k"]
            loss_kl_positive_wiki20k, _ = self(x, y, loss="kl")

            loss = (
                self.gamma * loss_ul_negated_wiki20k
                + (1 - self.gamma) * loss_kl_positive_wiki20k
            )

            x, y = val_batch["wiki20k"]
            loss_kl_wiki20k, _ = self(x, y, loss="kl")
            loss += loss_kl_wiki20k

            self.log_dict({"val_loss": loss}, prog_bar=True)
        else:
            x, y = val_batch
            loss, _ = self(x, y)
            self.log_dict({"val_loss": loss}, prog_bar=True, on_epoch=True)

    def predict(self, context: str, hypothesis: str = None, task="mlm", device="cpu"):
        """
        Make predictions using the PLBERTNOT model.

        Args:
            context (str): Input context.
            hypothesis (str): Input hypothesis (optional).
            task (str): Task type, either "mlm" (masked language modeling) or "te" (text entailment).
            device (str): Device to run the model on.

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model.predict(context, hypothesis, task, device)
