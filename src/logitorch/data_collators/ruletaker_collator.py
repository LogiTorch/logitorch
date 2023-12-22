from typing import Dict, Tuple

import torch
from transformers import RobertaTokenizer

from logitorch.datasets.proof_qa.proofwriter_dataset import PROOFWRITER_LABEL_TO_ID


class RuleTakerCollator:
    """
    A collator class for RuleTaker model.

    This collator is used to preprocess and collate data for RuleTaker model training or inference.

    Args:
        None

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing the batch inputs and labels.
    """

    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Preprocesses and collates the batch data.

        Args:
            batch: A list of tuples containing the context, question, label, and additional information.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing the batch inputs and labels.
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
    """
    A collator class for RuleTaker with ProofWriter model.

    This collator is used to preprocess and collate data for RuleTaker with ProofWriter model training or inference.

    Args:
        None

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing the batch inputs and labels.
    """

    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "LIAMF-USP/roberta-large-finetuned-race"
        )

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Preprocesses and collates the batch data.

        Args:
            batch: A list of tuples containing the context, question, label, and additional information.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing the batch inputs and labels.
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
