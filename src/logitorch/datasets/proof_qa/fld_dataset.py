from typing import Dict, List, Optional, Tuple, Union

from logitorch.datasets.base import AbstractProofQADataset
from logitorch.datasets.exceptions import (
    DatasetNameError,
    SplitSetError,
    TaskError,
)
from logitorch.datasets.utils import SPLIT_SETS
from datasets import load_dataset

FLD_SUB_DATASETS = [
    "FLD.v2",
    "FLD_star.v2",
]
FLD_TASKS = [
    "proof_generation_all",
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

            if dataset_name == "FLD.v2":
                hf_path, hf_name = "hitachi-nlp/FLD.v2", "default"
            elif dataset_name == "FLD_star.v2":
                hf_path, hf_name = "hitachi-nlp/FLD.v2", "star"
            hf_split = "validation" if split_set == "dev" else split_set
            hf_dataset = load_dataset(
                hf_path,
                name=hf_name,
                split=hf_split,
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
        Dict[str, Union[Optional[str], Optional[int]]],
    ]:
        return self._hf_dataset[index]

    def __str__(self) -> str:
        return f'The {self.split_set} set of {self.dataset_name}\'s FLD for the task of "{self.task}" has {self.__len__()} instances'

    def __len__(self) -> int:
        return len(self._hf_dataset)
