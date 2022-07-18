import torch.nn as nn
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


class RuleSelector(nn.Module):
    def __init__(self, pretrained_roberta_model: str, cls_dropout=0.1) -> None:
        super().__init__()
        self.model = RobertaModel.from_pretrained(pretrained_roberta_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_roberta_model)
        self.out_dim = self.model.config.hidden_size
        self.classifier = nn.Linear(self.out_dim, 1)
        self.dropout = nn.Dropout(cls_dropout)

        nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()

    def forward(self, x, y=None):
        last_hidden_state = self.model(**x)[0]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state).squeeze()

        return logits


class FactSelector(nn.Module):
    def __init__(self, pretrained_roberta_model: str) -> None:
        super().__init__()
        self.model = RobertaModel.from_pretrained(pretrained_roberta_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_roberta_model)
        self.out_dim = self.model.config.hidden_size
        self.classifier = nn.Linear(self.out_dim, 1)

        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)

        nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()

    def forward(self, x, y=None):
        pass


class KnowledgeComposer(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_model)


class FaiRR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
