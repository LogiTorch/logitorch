from typing import Dict, Tuple

import torch
from transformers import RobertaTokenizer

from logitorch.datasets.proof_qa.proofwriter_dataset import PROOFWRITER_LABEL_TO_ID


class RuleTakerCollator:
    def __init__(self) -> None:
        """
        The function __init__() is a constructor that initializes the tokenizer variable to the
        RobertaTokenizer.from_pretrained() function
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The function takes in a batch of data, and returns a tuple of two elements: the first element is
        a dictionary of tensors, and the second element is a tensor

        :param batch: A list of tuples of the form (context, question, label, depth)
        :return: A tuple of two tensors.
        """
        contexts = []
        questions = []
        batch_y = []

        for context, question, label, _ in batch:
            contexts.append(context)
            questions.append(question)
            batch_y.append(label)

        batch_x = self.tokenizer(contexts, questions, padding=True, return_tensors="pt")
        return batch_x, torch.tensor(batch_y, dtype=torch.int64)


class RuleTakerProofWriterCollator:
    def __init__(self) -> None:
        """
        The function __init__() is a constructor that initializes the tokenizer variable to the
        RobertaTokenizer.from_pretrained() function
        """
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        It takes a batch of data and returns a tuple of two tensors. The first tensor is a dictionary of
        tensors, and the second tensor is a tensor of labels.

        :param batch: A list of tuples of the form (context, question, label)
        :return: A tuple of two tensors.
        """
        contexts = []
        questions = []
        labels = []

        for i in batch:
            sentences = []
            for k, v in i[0].items():
                sentences.append(f"{k}: {v}")
            for k, v in i[1].items():
                sentences.append(f"{k}: {v}")

            contexts.append("".join(sentences))
            questions.append(i[2])
            labels.append(PROOFWRITER_LABEL_TO_ID[str(i[3])])

        batch_x = self.tokenizer(contexts, questions, padding=True, return_tensors="pt")
        return batch_x, torch.tensor(labels, dtype=torch.int64)
