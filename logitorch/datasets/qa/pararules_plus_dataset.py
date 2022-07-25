import os
from typing import List, Tuple

from logitorch.datasets.base import AbstractQADataset
from logitorch.datasets.exceptions import DatasetNameError, SplitSetError
from logitorch.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

PARARULES_PLUS_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/tynwrcyokf5vfa7/pararule_plus_dataset.zip?dl=1"
)
PARARULES_PLUS_DATASET = "pararules_plus_dataset"
PARARULES_PLUS_SUB_DATASETS = [
    "depth-2",
    "depth-3",
    "depth-4",
    "depth-5",
]
PARARULES_PLUS_DATASET_FOLDER = f"{DATASETS_FOLDER}/{PARARULES_PLUS_DATASET}"
PARARULES_PLUS_LABEL_TO_ID = {"false": 0, "true": 1}
PARARULES_PLUS_ID_TO_LABEL = {0: "False", 1: "True"}


class ParaRulesPlusDataset(AbstractQADataset):
    def __init__(self, dataset_name: str, split_set: str) -> None:
        super().__init__()
        try:
            if dataset_name not in PARARULES_PLUS_SUB_DATASETS:
                raise DatasetNameError()
            if split_set == "val":
                split_set = "dev"
            elif split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(PARARULES_PLUS_DATASET_FOLDER):
                download_dataset(PARARULES_PLUS_DATASET_ZIP_URL, PARARULES_PLUS_DATASET)

            self.dataset_name = dataset_name
            self.split_set = split_set
            self.dataset_path = f"{PARARULES_PLUS_DATASET_FOLDER}/{self.dataset_name}/{self.split_set}.jsonl"
            self.contexts, self.questions, self.labels = self.__read_dataset(
                "context", "questions", "text", "label"
            )
        except DatasetNameError as err:
            print(err.message)
            print(f"The ParaRules PLUS datasets are: {PARARULES_PLUS_SUB_DATASETS}")
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
                labels_list.append(PARARULES_PLUS_LABEL_TO_ID[str(q[str(labels_key)])])

        return contexts_list, questions_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self.contexts[index], self.questions[index], self.labels[index]

    def __str__(self) -> str:
        return f"The {self.split_set} set of {self.dataset_name}'s ParaRules PLUS has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.contexts)
