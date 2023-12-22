import torch.nn as nn
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


class RuleSelector(nn.Module):
    """
    RuleSelector is a class that represents a rule-based selector model.
    It uses a pretrained RoBERTa model for encoding input sequences and a linear classifier for prediction.
    """

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
        """
        Forward pass of the RuleSelector model.

        Args:
            x (dict): Input dictionary containing the input sequences.
            y (None): Placeholder for compatibility with other models.

        Returns:
            torch.Tensor: Logits representing the predicted scores.
        """
        last_hidden_state = self.model(**x)[0]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state).squeeze()

        return logits


class FactSelector(nn.Module):
    """
    FactSelector is a class that represents a fact-based selector model.
    It uses a pretrained RoBERTa model for encoding input sequences and a linear classifier for prediction.
    """

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
        """
        Forward pass of the FactSelector model.

        Args:
            x (dict): Input dictionary containing the input sequences.
            y (None): Placeholder for compatibility with other models.

        Returns:
            None: This method is not implemented.
        """
        pass


class KnowledgeComposer(nn.Module):
    """
    KnowledgeComposer is a class that represents a knowledge composer model.
    It uses a pretrained T5 model for generating text based on input prompts.
    """

    def __init__(self, pretrained_t5_model: str) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_model)


class FaiRR(nn.Module):
    """
    FaiRR is a class that represents the FaiRR model, which combines rule-based and fact-based selectors.
    """

    def __init__(self) -> None:
        super().__init__()
