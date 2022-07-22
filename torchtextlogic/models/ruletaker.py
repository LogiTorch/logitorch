from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.init import xavier_normal_
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaForMultipleChoice,
    RobertaModel,
    RobertaTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

# class RuleTaker(nn.Module):
#     def __init__(self, pretrained_model: str) -> None:
#         super().__init__()
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             pretrained_model, num_labels=2
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

#     def forward(
#         self, x: torch.Tensor, y: torch.Tensor = None
#     ) -> SequenceClassifierOutput:
#         if y is not None:
#             return self.model(**x, labels=y)
#         return self.model(**x)

#     def predict(self, context: str, question: str, device: str = "cpu") -> int:
#         with torch.no_grad():
#             tokenized_x = self.tokenizer(context, question, return_tensors="pt")
#             logits = self(tokenized_x.to(device)).logits
#             pred = logits.argmax().item()
#             return pred


class RuleTaker(nn.Module):
    def __init__(self, pretrained_roberta_model: str, num_labels: int = 2) -> None:
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
        with torch.no_grad():
            tokenized_x = self.tokenizer(context, question, return_tensors="pt")
            logits = self(tokenized_x.to(device)).logits
            pred = logits.argmax().item()
            return pred
