from typing import Dict, Tuple, List, Any, Tuple
import logging

import torch
from transformers import T5Tokenizer


logger = logging.getLogger(__name__)


class FLDProofGenerationAllCollator:
    def __init__(self,
                pretrained_t5_tokenizer: str,
                max_src_length=1024,
                max_tgt_length=512,
                log_examples=False) -> None:
        """
        Initializes the FLDProofGenerationAllCollator.

        Args:
            | pretrained_t5_tokenizer (str): The path or name of the pretrained T5 tokenizer.
            | max_src_length (int, optional): The maximum length of the source sequence. Defaults to 1024.
            | max_tgt_length (int, optional): The maximum length of the target sequence. Defaults to 512.
            | log_examples (bool, optional): Whether to log the examples during collation. Defaults to False.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_tokenizer)
        self.log_examples = log_examples
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def __call__(self, batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collates a batch of examples.

        Args:
            | batch (List[Dict]): A list of dictionaries representing the examples.

        Returns:
            | Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing the collated source sequences and target sequences.
        """
        xs: List[str] = []
        ys: List[str] = []
        for i_example, example in enumerate(batch):
            prompt = example['prompt_serial']
            gold_proof = example['proof_serial']

            x = prompt
            y = gold_proof
            xs.append(x)
            ys.append(y)
            if self.log_examples:
                logger.info('----------------- collate example = [%d]', i_example)
                logger.info('x     : "%s"', x)
                logger.info('y  : "%s"', y)
                logger.info('gold_proof : "%s"', gold_proof)

        batch_x = self.tokenizer(
            text=xs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_src_length,
            truncation=True,
        )
        batch_y = self.tokenizer(
            ys,
            padding=True,
            return_tensors="pt",
            max_length=self.max_tgt_length,
            truncation=True,
        )

        return batch_x, batch_y.input_ids
