import os
from typing import Any, Tuple

from torchtextlogic.datasets.abstract_dataset import AbstractQADataset
from torchtextlogic.datasets.utils import DATASETS_FOLDER_NAME, DATASETS_ZIP_FOLDER_NAME

DATASET_ZIP_URL = "https://www.dropbox.com/s/4j6jcc7ld5rf2tf/ruletaker_dataset.zip?dl=1"
RULETAKER_DATASETS= [
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

    def read_dataset(self, dataset_name: str) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        pass

    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        pass
