from typing import Dict, List, Optional, Tuple, Union, Any

from logitorch.datasets.base import AbstractProofQADataset
from logitorch.datasets.exceptions import (
    DatasetNameError,
    SplitSetError,
    TaskError,
)
from logitorch.datasets.utils import SPLIT_SETS
from datasets import load_dataset
from FLD_task import load_deduction, serialize
# from FLD_task.hf_dataset import serialize_transform

FLD_SUB_DATASETS = [
    "hitachi-nlp/FLD.v2",
    "hitachi-nlp/FLD_star.v2",
]
FLD_TASKS = [
    "proof_generation_all",
    # "proof_generation_iter",
    # "implication_enumeration",
    # "abduction",
]


class FLDDataset(AbstractProofQADataset):
    def __init__(
        self,
        dataset_name: str,
        split_set: str,
        task: str,
        max_samples: Optional[int] = None,
    ) -> None:
        try:
            if dataset_name not in FLD_SUB_DATASETS:
                raise DatasetNameError()

            if split_set == "val":
                split_set = "dev"
            elif split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if task not in FLD_TASKS:
                raise TaskError()

            self.dataset_name = dataset_name
            self.split_set = split_set
            self.task = task

            hf_split = 'validation' if split_set == 'dev' else split_set
            hf_dataset = load_dataset(
                dataset_name,
                split=hf_split,
            )

            # load and dump once to normalize dataset format to the latest version.
            hf_dataset = hf_dataset.map(
                lambda example: load_deduction(example).dict(),
                batched=False,
            )
            if max_samples is not None:
                hf_dataset = hf_dataset.select(range(max_samples))
            self._hf_dataset = hf_dataset

        except DatasetNameError as err:
            print(err.message)
            print(f"The FLD datasets are: {FLD_SUB_DATASETS}")
            raise err
        except SplitSetError as err:
            print(err.message)
            raise err
        except TaskError as err:
            print(err.message)
            print(f"The FLD tasks are: {FLD_TASKS}")
            raise err

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[
            Dict[str, str],
            Dict[str, str],
            List[str],
            List[str],
            List[str],
            List[str],
            List[int],
        ],
        Tuple[Dict[str, str], Dict[str, str], List[str], List[str], List[str]],
        Tuple[Dict[str, str], Dict[str, str], List[str], List[str]],
        Tuple[Dict[str, str], Dict[str, str], List[str]],

        Dict[str, Any],  # FLD dataset
    ]:
        hf_example = self._hf_dataset[index]
        deduction = load_deduction(hf_example)

        if self.task == "proof_generation_all":
            serial = serialize(
                deduction,
                stepwise=False,
                sample_negative_proof=False,
            )
        elif self.task == "proof_generation_iter":
            sample_negative_proof = self.split_set == 'train'
            serial = serialize(
                deduction,
                stepwise=True,
                sample_negative_proof=sample_negative_proof,
            )
        else:
            raise ValueError()

        hf_example_with_serial = hf_example.copy()
        hf_example_with_serial.update({
            'prompt_serial': serial.prompt,
            'partial_proof_serial': serial.partial_proof,
            'next_proof_step_serial': serial.next_proof_step,
            'proof_serial': serial.proofs[0],
        })
        return hf_example_with_serial

    def __str__(self) -> str:
        return f'The {self.split_set} set of {self.dataset_name}\'s FLD for the task of "{self.task}" has {self.__len__()} instances'

    def __len__(self) -> int:
        # return len(self.triples)
        return len(self._hf_dataset)
