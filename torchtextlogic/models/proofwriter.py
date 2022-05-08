from typing import Dict, List

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import SequenceClassifierOutput


class ProofWriter(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        """
        The function takes a pretrained T5 model and creates a T5ForConditionalGeneration object

        :param pretrained_t5_model: The name of the T5 model to use
        :type pretrained_t5_model: str
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)

    def forward(self, **x: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        """
        The function takes a dictionary of tensors as input and returns a SequenceClassifierOutput
        object

        :param : **x**: A dictionary of input tensors
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
