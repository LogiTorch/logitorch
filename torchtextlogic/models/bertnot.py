import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from losses.unlikelihood_loss import UnlikelihoodLoss
from models.exceptions import LossError, TaskError


class BERTNOT(nn.Module):
    def __init__(self, pretrained_bert_model: str, num_labels: int = 2) -> None:
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_bert_model)
        self.mlm_classifier = BertOnlyMLMHead(self.encoder.config)
        classifier_dropout = (
            self.encoder.config.classifier_dropout
            if self.encoder.config.classifier_dropout is not None
            else self.encoder.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.sequence_classifier = nn.Linear(
            self.encoder.config.hidden_size, num_labels
        )

        self.original_bert = BertForMaskedLM.from_pretrained(pretrained_bert_model)

        self.tasks = ["mlm", "te"]
        self.losses = ["cross_entropy", "unlikelihood", "kl"]

        self.num_labels = 2
        self.original_bert_softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.cross_entopy_loss = nn.CrossEntropyLoss()
        self.unlikelihood_loss = UnlikelihoodLoss()
        self.kl_loss = nn.KLDivLoss(reduction="sum")

    def forward(self, x, y=None, task="mlm", loss="cross_entropy"):
        try:
            if task not in self.tasks:
                raise TaskError(self.tasks)
            if loss not in self.losses:
                raise LossError(self.losses)

            if task == "mlm":
                outputs = self.encoder(**x)
                logits = self.mlm_classifier(outputs[0])

                if y is not None:
                    if loss == "cross_entropy":
                        loss = self.cross_entopy_loss(
                            logits.view(-1, self.encoder.config.vocab_size), y.view(-1)
                        )
                        return (loss, logits)
                    elif loss == "unlikelihood":
                        loss = self.unlikelihood_loss(
                            logits.view(-1, self.encoder.config.vocab_size), y.view(-1)
                        )
                        return (loss, logits)
                    else:
                        original_outputs = self.original_bert(**x)[0]
                        mask_token_indexes = torch.ne(y, -100)
                        original_outputs = original_outputs[mask_token_indexes]
                        original_probs = self.original_bert_softmax(original_outputs)

                        pred_probs = self.log_softmax(logits[mask_token_indexes])

                        loss = self.kl_loss(pred_probs, original_probs)

                        return (loss, logits)
                else:
                    return logits
            else:
                outputs = self.encoder(**x)
                sequence_outputs = self.dropout(outputs[0])
                logits = self.sequence_classifier(sequence_outputs)

                if y is not None:
                    loss = self.cross_entopy_loss(
                        logits.view(-1, self.num_labels), y.view(-1)
                    )
                    return (loss, logits)
                else:
                    return logits

        except TaskError as err:
            print(err.message)

    def predict(self, x: str, task="mlm", device="cpu"):
        try:
            if task != "mlm" or task != "te":
                raise TaskError()
        except TaskError as err:
            print(err.message)
