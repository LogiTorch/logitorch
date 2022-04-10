from typing import Dict, List

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutput


class ProofWriter(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)

    def forward(self, **x: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        return self.model(**x)

    def predict(self, **x: Dict[str, torch.Tensor]) -> List[str]:
        pass
