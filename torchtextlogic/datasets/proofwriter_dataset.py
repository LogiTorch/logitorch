import os
from typing import List, Tuple

from torchtextlogic.datasets.base_dataset import AbstractProofQADataset
from torchtextlogic.datasets.exceptions import DatasetNameError, SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

PROOFWRITER_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/5dpm2yjcm5h9hsd/proofwriter_dataset.zip?dl=1"
)
PROOFWRITER_DATASET = "proofwriter_dataset"
PROOFWRITER_SUB_DATASETS = [
    "birds-electricity",
    "depth-0",
    "depth-1",
    "depth-2",
    "depth-3",
    "depth-3ext",
    "depth-3ext-NatLang",
    "depth-5",
    "NatLang",
]
PROOFWRITER_DATASET_FOLDER = f"{DATASETS_FOLDER}/{PROOFWRITER_DATASET}"
PROOFWRITER_LABEL_TO_ID = {False: 0, True: 1}
PROOFWRITER_ID_TO_LABEL = {0: "False", 1: "True", 2: "Unknown"}


class ProofWriterDataset(AbstractProofQADataset):
    def __init__(
        self, dataset_name: str, split_set: str, open_world_assumption: bool = False
    ) -> None:
        pass

    def __read_dataset(self):
        pass

    def __getitem__(self, index) -> Tuple[str, str, str, int]:
        pass

    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        pass
