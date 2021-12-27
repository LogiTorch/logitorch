import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification


class RuleTaker(nn.Module):
    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=2
        )

    def forward(self, **x):
        return self.model(**x)
