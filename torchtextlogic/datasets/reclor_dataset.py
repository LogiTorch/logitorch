import os
from typing import Any, List, Optional, Tuple

from torchtextlogic.datasets.abstract_dataset import AbstractMCQADataset
from torchtextlogic.datasets.exceptions import SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

RECLOR_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/4dabc3ea0cf6sre/reclor_dataset.zip?dl=1"
)
RECLOR_DATASET = "reclor_dataset"
RECLOR_DATASET_FOLDER = f"{DATASETS_FOLDER}/{RECLOR_DATASET}"


class ReClorDataset(AbstractMCQADataset):
    def __init__(self, split_set: str) -> None:
        super().__init__()
        try:
            if split_set == "val":
                split_set = "dev"
            elif split_set not in SPLIT_SETS:
                raise SplitSetError()

            if not os.path.exists(RECLOR_DATASET_FOLDER):
                download_dataset(RECLOR_DATASET_ZIP_URL, RECLOR_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{RECLOR_DATASET_FOLDER}/{self.split_set}"
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(self) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[str, str, List[str], Optional[Any]]:
        pass

    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        pass
