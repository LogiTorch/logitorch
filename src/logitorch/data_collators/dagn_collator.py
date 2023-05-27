from typing import Dict, List, Tuple

import torch
from transformers import RobertaTokenizer


class DAGNCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_tokenizer)

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


def find_explicit_connectives(context: str, question: str, answers_list: List[str]):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    tokenized_context = tokenizer.tokenize(context)
    tokenized_question = tokenizer.tokenize(question)
    tokenized_answers_list = []
    for ans in answers_list:
        tokenized_answers_list.append(tokenizer.tokenize(ans))

    print(tokenized_context)
    print(tokenized_question)
    print(tokenized_answers_list)


find_explicit_connectives("<s> Hello world</s>", "I play football", ["youpi </s>"])
# def __find_puncts():
#     pass
