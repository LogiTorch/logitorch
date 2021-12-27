import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification


class RuleTaker(pl.LightningModule):
    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=2
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, **x):
        return self.model(**x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self(**x)
        y_pred = outputs.logits
        loss = self.cross_entropy_loss(y_pred, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self(**x)
        y_pred = outputs.logits
        loss = self.cross_entropy_loss(y_pred, y)
        self.log("val_loss", loss, on_epoch=True)
