import torch.nn as nn
from transformers import RobertaModel


class PRover(nn.Module):
    def __init__(self, pretrained_roberta_model: str) -> None:
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_roberta_model)
