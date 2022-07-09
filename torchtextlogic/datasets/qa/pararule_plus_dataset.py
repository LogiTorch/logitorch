import os
from typing import List, Tuple

from torchtextlogic.datasets.base import AbstractQADataset
from torchtextlogic.datasets.exceptions import DatasetNameError, SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

PARARULE_PLUS_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/tynwrcyokf5vfa7/pararule_plus_dataset.zip?dl=1"
)
PARARULE_PLUS_DATASET = "pararule_plus_dataset"
PARARULE_PLUS_SUB_DATASETS = [
    "depth-2",
    "depth-3",
    "depth-4",
    "depth-5",
]
PARARULE_PLUS_DATASET_FOLDER = f"{DATASETS_FOLDER}/{PARARULE_PLUS_DATASET}"
PARARULE_PLUS_LABEL_TO_ID = {"false": 0, "true": 1}
PARARULE_PLUS_ID_TO_LABEL = {0: "False", 1: "True"}


class PararulePlusDataset(AbstractQADataset):
    def __init__(self, dataset_name: str, split_set: str) -> None:
        super().__init__()
        try:
            if dataset_name not in PARARULE_PLUS_SUB_DATASETS:
                raise DatasetNameError()
            if split_set == "val":
                split_set = "dev"
            elif split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(PARARULE_PLUS_DATASET_FOLDER):
                download_dataset(PARARULE_PLUS_DATASET_ZIP_URL, PARARULE_PLUS_DATASET)

            self.dataset_name = dataset_name
            self.split_set = split_set
            self.dataset_path = f"{PARARULE_PLUS_DATASET_FOLDER}/{self.dataset_name}/{self.split_set}.jsonl"
            self.contexts, self.questions, self.labels = self.__read_dataset(
                "context", "questions", "text", "label"
            )
        except DatasetNameError as err:
            print(err.message)
            print(f"The Pararule Plus datasets are: {PARARULE_PLUS_SUB_DATASETS}")
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self,
        contexts_key: str,
        questions_key: str,
        questions_text_key: str,
        labels_key: str,
    ) -> Tuple[List[str], List[str], List[int]]:
        data = read_jsonl(self.dataset_path)
        contexts_list = []
        questions_list = []
        labels_list = []

        for i in data:
            for q in i[questions_key]:
                contexts_list.append(i[contexts_key])
                questions_list.append(q[questions_text_key])
                labels_list.append(PARARULE_PLUS_LABEL_TO_ID[str(q[str(labels_key)])])

        return contexts_list, questions_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self.contexts[index], self.questions[index], self.labels[index]

    def __str__(self) -> str:
        return f"The {self.split_set} set of {self.dataset_name}'s Pararule Plus has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.contexts)
