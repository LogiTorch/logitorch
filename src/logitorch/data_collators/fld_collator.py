from typing import Dict, Tuple, List, Any, Tuple
import logging
import random

import torch
from transformers import T5Tokenizer


logger = logging.getLogger(__name__)


class FLDProofGenerationAllCollator:
    def __init__(self,
                 pretrained_t5_tokenizer: str,
                 max_src_length=1024,
                 max_tgt_length=512,
                 log_examples=True) -> None:
        """
        It takes in a pretrained T5 tokenizer

        :param pretrained_t5_tokenizer: The name of the T5 tokenizer to use
        :type pretrained_t5_tokenizer: str
        """
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_tokenizer)
        self.log_examples = log_examples
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __call__(self, batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        The function takes in a batch of data, and returns a tuple of two elements: a dictionary of
        tensors, and a tensor

        :param batch: A list of tuples of the form (context, question, label, proof)
        :return: A tuple of two tensors. The first tensor is the input tensor, and the second tensor is
        the output tensor.
        """

        prompts: List[str] = []
        proof_steps: List[str] = []
        for i_example, example in enumerate(batch):
            prompt = example['context']
            next_step = example['next_step']
            gold_proof = example['gold_proof']

            prompts.append(prompt)
            proof_steps.append(next_step)
            if self.log_examples:
                logger.info('----------------- collate example = [%d]', i_example)
                logger.info('prompt     : "%s"', prompt)
                logger.info('next_step  : "%s"', next_step)
                logger.info('gold_proof : "%s"', gold_proof)

        batch_x = self.tokenizer(
            text=prompts,
            padding=True,
            return_tensors="pt",
            max_length=self.max_src_length,
            truncation=True,
        )
        batch_y = self.tokenizer(
            proof_steps,
            padding=True,
            return_tensors="pt",
            max_length=self.max_tgt_length,
            truncation=True,
        )

        return batch_x, batch_y.input_ids
