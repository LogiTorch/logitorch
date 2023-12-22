import torch.nn as nn
from transformers import AutoConfig, RobertaConfig


class DAGN(nn.Module):
    def __init__(self, pretrained_roberta_model) -> None:
        super().__init__()
        self.roberta_model = RobertaConfig.from_pretrained(pretrained_roberta_model)
