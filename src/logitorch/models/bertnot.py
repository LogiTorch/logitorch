import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from transformers import BertForMaskedLM, BertTokenizer

from logitorch.losses.unlikelihood_loss import UnlikelihoodLoss
from logitorch.models.exceptions import LossError, TaskError


class BERTNOT(nn.Module):
    def __init__(self, pretrained_bert_model: str, num_labels: int = 2) -> None:
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained(pretrained_bert_model)

        classifier_dropout = (
            self.model.config.classifier_dropout
            if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.sequence_classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        self.original_bert = BertForMaskedLM.from_pretrained(pretrained_bert_model)

        self.tasks = ["mlm", "te"]
        self.losses = ["cross_entropy", "unlikelihood", "kl"]

        self.num_labels = num_labels
        self.original_bert_softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.cross_entopy_loss = nn.CrossEntropyLoss()
        self.unlikelihood_loss = UnlikelihoodLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_model)

        xavier_normal_(self.sequence_classifier.weight)

    def forward(self, x, y=None, task="mlm", loss="cross_entropy"):
        try:
            if task not in self.tasks:
                raise TaskError(self.tasks)
            if loss not in self.losses:
                raise LossError(self.losses)

            if task == "mlm":
                outputs = self.model(**x)
                logits = outputs.logits

                if y is not None:
                    if loss == "cross_entropy":
                        loss = self.cross_entopy_loss(
                            logits.view(-1, self.model.config.vocab_size), y.view(-1)
                        )
                        return (loss, logits)
                    elif loss == "unlikelihood":
                        loss = self.unlikelihood_loss(
                            logits.view(-1, self.model.config.vocab_size), y.view(-1)
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
                outputs = self.model.bert(**x)[0]
                cls_representation = outputs[:, 0, :]
                sequence_outputs = self.dropout(cls_representation)
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

    def predict(self, context: str, hypothesis: str = None, task="mlm", device="cpu"):
        try:
            if task not in self.tasks:
                raise TaskError(self.tasks)

            if hypothesis is None:
                tokenized_x = self.tokenizer(context, return_tensors="pt")
            else:
                tokenized_x = self.tokenizer(context, hypothesis, return_tensors="pt")

            logits = self(tokenized_x.to(device), task=task)
            if task == "mlm":
                mask_token_indexes = (
                    tokenized_x.input_ids == self.tokenizer.mask_token_id
                )[0].nonzero(as_tuple=True)[0]
                predicted_token_id = logits[0, mask_token_indexes].argmax(axis=-1)
                return self.tokenizer.decode(predicted_token_id)
            else:
                pred = logits.argmax().item()
                return pred
        except TaskError as err:
            print(err.message)
