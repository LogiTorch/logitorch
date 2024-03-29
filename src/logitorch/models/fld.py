from typing import Dict, List

import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class FLDAllAtOnceProver(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        """
        Initializes the FLDAllAtOnceProver model.

        Args:
            pretrained_t5_model (str): The name or path of the pretrained T5 model.
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_model)

    def forward(
        self, x: Dict[str, torch.Tensor], y: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        """
        Performs a forward pass of the model.

        Args:
            x (Dict[str, torch.Tensor]): The input tensors.
            y (torch.Tensor, optional): The labels tensor. Defaults to None.

        Returns:
            SequenceClassifierOutput: The output of the model.
        """
        if y is not None:
            return self.model(**x, labels=y)
        return self.model(**x)

    def predict(
        self,
        prompt: str,
        num_beams: int = 5,
        max_length: int = 1000,
        device: str = "cpu",
    ) -> List[str]:
        """
        Generates predictions based on the given prompt.

        Args:
            prompt (str): The input prompt.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            max_length (int, optional): The maximum length of the generated sequence. Defaults to 1000.
            device (str, optional): The device to run the model on. Defaults to "cpu".

        Returns:
            List[str]: The generated predictions.
        """
        with torch.no_grad():
            tokenized_x = self.tokenizer(
                prompt, padding=True, return_tensors="pt"
            )

            beam_output = self.model.generate(
                **tokenized_x.to(device),
                max_length=max_length,
                num_beams=num_beams,
                do_sample=True,
                top_p=0.90
            )
            pred = self.tokenizer.decode(
                beam_output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        return pred
