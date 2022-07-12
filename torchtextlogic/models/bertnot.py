import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


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

    def forward(self, x, y=None, task="mlm", loss=None):
        pass

    def predict(self, x, task="mlm"):
        pass


model = BERTNOT("bert-base-uncased")
