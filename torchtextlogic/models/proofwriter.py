import torch.nn as nn
from transformers import T5ForConditionalGeneration


class ProofWriter(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)

    def forward(self, **x):
        return self.model(**x)
