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
        context: str,
        question: str,
        num_beams: int = 5,
        max_length: int = 120,
        device: str = "cpu",
    ) -> List[str]:
        with torch.no_grad():
            tokenized_x = self.tokenizer(
                context, question, padding=True, return_tensors="pt"
            )

            beam_output = self.model.generate(
                **tokenized_x.to(device),
                max_length=max_length,
                num_beams=num_beams,
                do_sample=True,
                top_p=0.90
            )
            # beam_output = self.model.generate(
            #     **tokenized_x.to(device),
            #     max_length=max_length,
            #     early_stopping=True,
            #     num_beams=num_beams,
            #     num_return_sequences=1,
            #     use_cache=True,
            #     no_repeat_ngram_size=1,
            #     remove_invalid_values=True,
            # )
            # print(beam_output)
            pred = self.tokenizer.decode(
                beam_output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        return pred
