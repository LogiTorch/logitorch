from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer


class DAGNCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        contexts = []
        questions = []
        batch_y = []

        for context, question, answer_options, label in batch:
            for option in answer_options:
                contexts.append(context)
                questions.append(f"{question} {option}")
                batch_y.append(label)

        batch_x = self.tokenizer(contexts, questions, padding=True, return_tensors="pt")
        return batch_x, torch.tensor(batch_y, dtype=torch.int64)
