import torch.nn as nn
from transformers import AutoConfig, RobertaConfig


class DAGN(nn.Module):
    def __init__(self, pretrained_roberta_model) -> None:
        """
        This function takes in a pretrained RoBERTa model and returns a RobertaConfig object

        :param pretrained_roberta_model: The name of the pretrained RoBERTa model to use
        """
        super().__init__()
        self.roberta_model = RobertaConfig.from_pretrained(pretrained_roberta_model)
