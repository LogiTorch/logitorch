from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class LReasoner(nn.Module):
    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=4
        )

    def forward(self, **x: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        return self.model(**x)

    def predict(self, **x: Dict[str, torch.Tensor]) -> List[str]:
        pass
