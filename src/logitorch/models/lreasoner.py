from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class LReasoner(nn.Module):
    def __init__(self, pretrained_model: str) -> None:
        """
        Initializes an instance of the LReasoner class.

        Args:
            pretrained_model (str): The name or path of the pre-trained model to use.

        Returns:
            None
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=4
        )

    def forward(self, **x: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        """
        Performs forward pass through the LReasoner model.

        Args:
            **x (Dict[str, torch.Tensor]): Input tensors for the model.

        Returns:
            SequenceClassifierOutput: The output of the model.
        """
        return self.model(**x)

    def predict(self, **x: Dict[str, torch.Tensor]) -> List[str]:
        """
        Predicts the class labels for the input tensors.

        Args:
            **x (Dict[str, torch.Tensor]): Input tensors for prediction.

        Returns:
            List[str]: The predicted class labels.
        """
        pass
