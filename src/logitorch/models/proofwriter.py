from typing import Dict, List

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


class ProofWriter(nn.Module):
    """
    A PyTorch module for generating proofs using the T5 model.

    Args:
        pretrained_t5_model (str): The name or path of the pretrained T5 model.

    Attributes:
        model (T5ForConditionalGeneration): The T5 model for proof generation.
        tokenizer (T5Tokenizer): The tokenizer for tokenizing input text.

    Methods:
        forward(x, y=None): Performs forward pass of the model.
        predict(context, question, num_beams=5, max_length=120, device="cpu"): Generates proof given context and question.
    """

    def __init__(self, pretrained_t5_model: str) -> None:
        """
        Initializes the ProofWriter module.

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
        Performs forward pass of the model.

        Args:
            |x (Dict[str, torch.Tensor]): The input tensors for the model.
            |y (torch.Tensor, optional): The labels for the model. Defaults to None.

        Returns:
            |SequenceClassifierOutput: The output of the model.
        """
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
        """
        Generates proof given context and question.

        Args:
            context (str): The context for proof generation.
            question (str): The question for proof generation.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            max_length (int, optional): The maximum length of the generated proof. Defaults to 120.
            device (str, optional): The device to run the model on. Defaults to "cpu".

        Returns:
            List[str]: The generated proof.
        """
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
            pred = self.tokenizer.decode(
                beam_output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        return pred
