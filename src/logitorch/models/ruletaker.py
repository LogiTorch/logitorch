from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForMultipleChoice, RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


class RuleTaker(nn.Module):
    """
    RuleTaker is a PyTorch module for rule-based question answering using the Roberta model.

    Args:
        num_labels (int): The number of labels for classification. Default is 2.

    Attributes:
        num_labels (int): The number of labels for classification.
        encoder (RobertaForMultipleChoice): The Roberta model for multiple choice tasks.
        config (RobertaConfig): The configuration of the Roberta model.
        classifier (RobertaClassificationHead): The classification head of the Roberta model.
        tokenizer (RobertaTokenizer): The tokenizer for the Roberta model.

    Methods:
        forward(x, y=None): Performs forward pass of the RuleTaker model.
        predict(context, question, device="cpu"): Predicts the answer label for a given context and question.

    """

    def __init__(self, num_labels: int = 2) -> None:
        """
        Initializes a RuleTaker instance.

        Args:
            num_labels (int): The number of labels for classification. Default is 2.

        """
        super().__init__()
        self.num_labels = num_labels
        self.encoder = RobertaForMultipleChoice.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        ).roberta
        self.config = self.encoder.config
        self.classifier = RobertaClassificationHead(self.config)
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )

    def forward(self, x, y=None):
        """
        Performs forward pass of the RuleTaker model.

        Args:
            x (dict): The input dictionary containing the context and question.
            y (Tensor): The target labels. Default is None.

        Returns:
            outputs (tuple): A tuple containing the logits and other outputs.

        """
        outputs = self.encoder(**x)
        sequence_outputs = outputs[0]
        logits = self.classifier(sequence_outputs)
        outputs = (logits,) + outputs[2:]

        if y is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), y.view(-1))
            outputs = (loss,) + outputs

        return outputs

    def predict(self, context: str, question: str, device: str = "cpu") -> int:
        """
        Predicts the answer label for a given context and question.

        Args:
            context (str): The context for the question.
            question (str): The question to be answered.
            device (str): The device to run the prediction on. Default is "cpu".

        Returns:
            pred (int): The predicted answer label.

        """
        with torch.no_grad():
            tokenized_x = self.tokenizer(context, question, return_tensors="pt")
            logits = self(tokenized_x.to(device))[0]
            pred = logits.argmax().item()
            return pred
