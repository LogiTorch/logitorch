import os
from typing import Any, Tuple

from torch.utils import data
from torchtextlogic.datasets.abstract_dataset import AbstractQADataset
from torchtextlogic.datasets.exceptions import DatasetNameError, SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER_NAME,
    DATASETS_ZIP_FOLDER_NAME,
    SPLIT_SETS,
)

DATASET_ZIP_URL = "https://www.dropbox.com/s/4j6jcc7ld5rf2tf/ruletaker_dataset.zip?dl=1"
RULETAKER_DATASETS = [
    "bird_electricity",
    "depth-0",
    "depth-1",
    "depth-2",
    "depth-3",
    "depth-3ext",
    "depth-3ext-NatLang",
    "depth-5",
    "NatLang",
]


class RuleTakerDataset(AbstractQADataset):
    def __init__(self, dataset_name: str, split_set: str) -> None:
        super().__init__()
        try:
            if dataset_name not in RULETAKER_DATASETS:
                raise DatasetNameError()
            if split_set not in SPLIT_SETS:
                raise SplitSetError()
        except DatasetNameError as err:
            print(err.message)
            print(f"The RuleTaker datasets are: {RULETAKER_DATASETS}")
        except SplitSetError as err:
            print(err.message)

        self.dataset_name = dataset_name
        self.split_set = split_set

    def read_dataset(self, dataset_name: str) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        pass

    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        pass
