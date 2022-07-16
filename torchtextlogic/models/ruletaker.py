from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class RuleTaker(nn.Module):
    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        if y is not None:
            return self.model(**x, labels=y)
        return self.model(**x)

    def predict(self, x: str, device: str = "cpu") -> int:
        with torch.no_grad():
            tokenized_x = self.tokenizer(x, return_tensors="pt")
            logits = self(tokenized_x.to(device)).logits
            pred = logits.argmax().item()
            return pred
