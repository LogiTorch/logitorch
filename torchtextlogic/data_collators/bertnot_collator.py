from typing import Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class BERTNOTWiki20KCollator:
    def __init__(self, pretrained_tokenizer: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)

    def __call__(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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
