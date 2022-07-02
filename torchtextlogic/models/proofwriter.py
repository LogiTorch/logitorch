from typing import Dict, List

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class ProofWriter(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_model)

    def forward(
        self, x: Dict[str, torch.Tensor], y: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        if y is not None:
            return self.model(**x, labels=y)
        return self.model(**x)

    def predict(
        self,
        x: str,
        prompt: str = None,
        num_beams: int = 5,
        max_length: int = 120,
        device: str = "cpu",
    ) -> List[str]:
        with torch.no_grad():
            if prompt is None:
                tokenized_x = self.tokenizer(x, padding=True, return_tensors="pt")
            else:
                new_x = [f"{prompt}: {i}" for i in x]
                tokenized_x = self.tokenizer(new_x, paddding=True, return_tensors="pt")

            beam_output = self.model.generate(
                **tokenized_x.to(device),
                max_length=max_length,
                early_stopping=True,
                num_beams=num_beams,
                num_return_sequences=1,
                use_cache=True,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )

            output = self.tokenizer.decode(
                beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        return output
