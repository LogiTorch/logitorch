import os
from typing import List, Tuple

from torchtextlogic.datasets.base_dataset import AbstractTEDataset
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
CONTROL_DATASET = "control_dataset"
CONTROL_DATASET_FOLDER = f"{DATASETS_FOLDER}/{CONTROL_DATASET}"
CONTROL_LABEL_TO_ID = {"c": 0, "e": 1, "n": 2}
CONTROL_ID_TO_LABEL = {0: "c", 1: "e", 2: "n"}


class ControlDataset(AbstractTEDataset):
    def __init__(self, split_set: str) -> None:
        super().__init__()

        try:
            if split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(CONTROL_DATASET_FOLDER):
                download_dataset(CONTROL_DATASET_ZIP_URL, CONTROL_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{CONTROL_DATASET_FOLDER}/{self.split_set}.jsonl"
            self.premises, self.hypotheses, self.labels = self.__read_dataset(
                "premise", "hypothesis", "label"
            )
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self, premises_key: str, hypotheses_key: str, labels_key: str
    ) -> Tuple[List[str], List[str], List[int]]:
        data = read_jsonl(self.dataset_path)
        premises_list = []
        hypotheses_list = []
        labels_list = []

        for i in data:
            premises_list.append(i[premises_key])
            hypotheses_list.append(i[hypotheses_key])
            labels_list.append(CONTROL_LABEL_TO_ID[i[labels_key]])

        return premises_list, hypotheses_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self.premises[index], self.hypotheses[index], self.labels[index]

    def __str__(self) -> str:
        return f"The {self.split_set} set of ConTRoL has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.premises)
