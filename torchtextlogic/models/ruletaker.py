from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class RuleTaker(nn.Module):
    def __init__(self, pretrained_model: str) -> None:
        """
        This function takes in a pretrained model and returns a model with the number of labels set to 2

        :param pretrained_model: str = "bert-base-uncased"
        :type pretrained_model: str
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=2
        )

    def forward(self, **x: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        """
        The function takes a dictionary of tensors as input and returns a SequenceClassifierOutput
        object

        :param :
        :type : Dict[str, torch.Tensor]
        :return: The model is being returned.
        """
        return self.model(**x)

    def predict(self, **x: Dict[str, torch.Tensor]) -> List[str]:
        """
        The function takes a dictionary of tensors as input and returns a list of strings

        :param : **x**: A dictionary of input tensors
        :type : Dict[str, torch.Tensor]
        """
        pass
