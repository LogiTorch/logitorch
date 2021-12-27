import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from torchtextlogic.models.ruletaker import RuleTaker


class PLRuleTaker(pl.LightningModule):
    def __init__(self, pretrained_model: str, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.model = RuleTaker(pretrained_model)
        self.learning_rate = learning_rate
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, **x):
        return self.model(**x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

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

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self(**x)
        y_pred = outputs.logits
        loss = self.cross_entropy_loss(y_pred, y)
        self.log("val_loss", loss, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        return outputs.logits
