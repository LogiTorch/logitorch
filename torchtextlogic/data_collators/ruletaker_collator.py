from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer

from torchtextlogic.datasets.proof_qa.proofwriter_dataset import PROOFWRITER_LABEL_TO_ID


class RuleTakerCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        contexts = []
        questions = []
        batch_y = []

        for context, question, label in batch:
            contexts.append(context)
            questions.append(question)
            batch_y.append(label)

        batch_x = self.tokenizer(contexts, questions, padding=True, return_tensors="pt")
        return batch_x, torch.tensor(batch_y, dtype=torch.int64)


class RuleTakerProofWriterCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        contexts = []
        questions = []
        labels = []

        for i in batch:
            sentences = []
            for s in i[0].values():
                sentences.append(s)
            for s in i[1].values():
                sentences.append(s)

        contexts.append("".join(sentences))
        questions.append(i[2])
        labels.append(PROOFWRITER_LABEL_TO_ID[str(i[3])])

        batch_x = self.tokenizer(contexts, questions, padding=True, return_tensors="pt")
        return batch_x, torch.tensor(labels, dtype=torch.int64)
