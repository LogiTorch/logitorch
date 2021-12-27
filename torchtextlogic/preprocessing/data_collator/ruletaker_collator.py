import torch
from transformers import AutoTokenizer


class RuleTakerCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __call__(self, batch):
        contexts = []
        questions = []
        batch_y = []

        for context, question, label in batch:
            contexts.append(context)
            questions.append(question)
            batch_y.append(label)

        batch_x = self.tokenizer(contexts, questions, padding=True, return_tensors="pt")
        return batch_x, torch.tensor(batch_y, dtype=torch.int64)
