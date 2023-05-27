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

    def forward(self, x, y=None, loss="cross_entropy"):  # type: ignore
        if self.task == "mlm":
            if y is not None:
                return self.model(x, y, task="mlm", loss=loss)
            return self.model(x, task="mlm")
        else:
            if y is not None:
                return self.model(x, y, task="te")
            return self.model(x, task="te")

    def configure_optimizers(self):

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

    def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int):  # type: ignore

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

    def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
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
        return self.model.predict(context, hypothesis, task, device)


# class PLBERTNOT(pl.LightningModule):
#     def __init__(
#         self,
#         load_path_model: str = None,
#         pretrained_model: str = "bert-base-uncased",
#         num_labels=2,
#         learning_rate: float = 2e-5,
#         batch_size: int = 32,
#     ) -> None:
#         super().__init__()
#         if load_path_model is not None:
#             self.model = torch.load(load_path_model)
#         else:
#             self.model = BERTNOT(pretrained_model, num_labels)
#         self.num_labels = num_labels
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size

#     def forward(self, x, y=None):  # type: ignore
#         if y is not None:
#             return self.model(x, y, task="te")
#         return self.model(x, y, task="te")

#     def predict(self, context: str, hypothesis: str = None, task="mlm", device="cpu"):
#         return self.model.predict(context, hypothesis, task, device)

#     def configure_optimizers(self):
#         paramaters = [
#             {"params": self.model.model.parameters()},
#             {"params": self.model.sequence_classifier.parameters()},
#         ]
#         return Adam(paramaters, lr=self.learning_rate)

#     def training_step(self, train_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int):  # type: ignore
#         x, y = train_batch
#         loss, _ = self(x, y)
#         self.log_dict({"train_loss": loss}, prog_bar=True, on_epoch=True)
#         return loss

#     def validation_step(self, val_batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> None:  # type: ignore
#         x, y = val_batch
#         loss, _ = self(x, y)
#         self.log_dict({"val_loss": loss}, prog_bar=True, on_epoch=True)
