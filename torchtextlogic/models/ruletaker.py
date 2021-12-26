import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizer


class RuleTaker(pl.LightningModule):
    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=2
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = nn.CrossEntropyLoss(y_pred, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        loss = nn.CrossEntropyLoss(y_pred, y)
        self.log("val_loss", loss, on_epoch=True)


def ruletaker_collate_fn(pretrained_tokenizer: PreTrainedTokenizer):
    pass
