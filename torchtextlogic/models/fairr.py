import torch.nn as nn
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


class RuleSelector(nn.Module):
    def __init__(self, pretrained_roberta_model: str) -> None:
        super().__init__()
        self.model = RobertaModel.from_pretrained(pretrained_roberta_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_roberta_model)


class FactSelector(nn.Module):
    def __init__(self, pretrained_roberta_model: str) -> None:
        super().__init__()
        self.model = RobertaModel.from_pretrained(pretrained_roberta_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_roberta_model)


class KnowledgeComposer(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_model)


class FaiRR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
