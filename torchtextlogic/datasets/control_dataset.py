import os
from typing import Any, Tuple

from torchtextlogic.datasets.abstract_dataset import AbstractTEDataset
from torchtextlogic.datasets.exceptions import SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

CONTROL_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/rmcqituydxacuhv/control_dataset.zip?dl=1"
)
CONTROL_DATASET = "reclor_dataset"
CONTROL_DATASET_FOLDER = f"{DATASETS_FOLDER}/{CONTROL_DATASET}"


class ControlDataset(AbstractTEDataset):
    def __init__(self) -> None:
        super().__init__()

    def __read_dataset(self) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        pass

    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        pass
