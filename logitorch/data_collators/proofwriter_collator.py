from typing import Dict, Tuple

import torch
from transformers import T5Tokenizer


class ProofWriterQACollator:
    def __init__(self, pretrained_t5_tokenizer: str) -> None:
        """
        It takes in a pretrained T5 tokenizer

        :param pretrained_t5_tokenizer: The name of the T5 tokenizer to use
        :type pretrained_t5_tokenizer: str
        """
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The function takes in a batch of data, and returns a tuple of two elements: a dictionary of
        tensors, and a tensor

        :param batch: A list of tuples of the form (context, question, label)
        :return: A tuple of two tensors. The first tensor is the input tensor, and the second tensor is
        the output tensor.
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
            labels.append(str(i[3]))

        batch_x = self.tokenizer(
            contexts,
            questions,
            padding=True,
            return_tensors="pt",
        )
        batch_y = self.tokenizer(labels, padding=True, return_tensors="pt")

        return batch_x, batch_y.input_ids


class ProofWriterProofGenerationAllCollator:
    def __init__(self, pretrained_t5_tokenizer: str) -> None:
        """
        It takes in a pretrained T5 tokenizer

        :param pretrained_t5_tokenizer: The name of the T5 tokenizer to use
        :type pretrained_t5_tokenizer: str
        """
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The function takes in a batch of data, and returns a tuple of two elements: a dictionary of
        tensors, and a tensor

        :param batch: A list of tuples of the form (context, question, label, proof)
        :return: A tuple of two tensors. The first tensor is the input tensor, and the second tensor is
        the output tensor.
        """
        contexts = []
        questions = []
        labels = []
        proofs = []

        for i in batch:
            sentences = []
            for k, v in i[0].items():
                sentences.append(f"{k}: {v}")
            for k, v in i[1].items():
                sentences.append(f"{k}: {v}")

            contexts.append("".join(sentences))
            questions.append(i[2])
            labels.append(str(i[3]))

            proof = i[4].split("OR")[0]
            proof = proof.replace("[", "")
            proof = proof.replace("]", "")
            proofs.append(proof)

        batch_x = self.tokenizer(
            contexts,
            questions,
            padding=True,
            return_tensors="pt",
        )
        batch_y = self.tokenizer(
            labels,
            proofs,
            padding=True,
            return_tensors="pt",
        )

        return batch_x, batch_y.input_ids
