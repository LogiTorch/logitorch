from typing import Dict, Tuple

import torch
from logitorch.datasets.proof_qa.proofwriter_dataset import PROOFWRITER_LABEL_TO_ID
from transformers import AutoTokenizer, RobertaTokenizer


class RuleTakerCollator:
    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )

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
    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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
