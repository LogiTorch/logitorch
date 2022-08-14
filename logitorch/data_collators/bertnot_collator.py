from typing import Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class BERTNOTWiki20KCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        """
        It takes in a pretrained tokenizer

        :param pretrained_tokenizer: The name of the pretrained tokenizer to use
        :type pretrained_tokenizer: str
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The function takes in a batch of data, and returns a tuple of two elements: a dictionary of
        tensors, and a tensor

        :param batch: A list of tuples of the form (sentence, label)
        :return: A tuple of two tensors. The first tensor is the input tensor, and the second tensor is
        the output tensor.
        """
        sentences = []
        labels = []

        for sentence, label in batch:
            tokenized_sentence = self.tokenizer.encode(sentence, return_tensors="pt")
            tokenized_label = self.tokenizer.encode(label, return_tensors="pt")

            if len(tokenized_sentence[0]) == len(tokenized_label[0]):
                mask_token_label_index = torch.where(
                    tokenized_sentence == self.tokenizer.mask_token_id,
                    tokenized_label,
                    -100,
                )
                sentences.append(sentence)
                labels.append(mask_token_label_index[0])

        batch_x = self.tokenizer(
            sentences,
            padding=True,
            return_tensors="pt",
        )

        batch_y = pad_sequence(labels, batch_first=True, padding_value=-100)

        return batch_x, batch_y


class BERTNOTTextualEntailmentCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        """
        It takes in a pretrained tokenizer

        :param pretrained_tokenizer: The name of the pre-trained tokenizer to use
        :type pretrained_tokenizer: str
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The function takes in a batch of data, and returns a tuple of two elements: a dictionary of
        tensors, and a tensor

        :param batch: A list of tuples of the form (premise, hypothesis, label)
        :return: A tuple of two tensors. The first tensor is the input tensor, and the second tensor is
        the output tensor.
        """
        premises = []
        hypotheses = []
        labels = []

        for p, h, l in batch:
            premises.append(p)
            hypotheses.append(h)
            labels.append(l)

        batch_x = self.tokenizer(
            premises, hypotheses, padding=True, return_tensors="pt"
        )
        batch_y = torch.tensor(labels)

        return batch_x, batch_y
