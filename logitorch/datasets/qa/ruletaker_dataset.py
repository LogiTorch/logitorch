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

RULETAKER_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/1ix9j86nlgl0ygp/ruletaker_dataset.zip?dl=1"
)
RULETAKER_DATASET = "ruletaker_dataset"
RULETAKER_SUB_DATASETS = [
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
RULETAKER_DATASET_FOLDER = f"{DATASETS_FOLDER}/{RULETAKER_DATASET}"
RULETAKER_LABEL_TO_ID = {"False": 0, "True": 1}
RULETAKER_ID_TO_LABEL = {0: "False", 1: "True"}


class RuleTakerDataset(AbstractQADataset):
    def __init__(self, dataset_name: str, split_set: str) -> None:
        super().__init__()
        try:
            if dataset_name not in RULETAKER_SUB_DATASETS:
                raise DatasetNameError()
            if split_set != "test" and dataset_name == "birds-electricity":
                raise SplitSetError(["test"])
            if split_set == "val":
                split_set = "dev"
            elif split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(RULETAKER_DATASET_FOLDER):
                download_dataset(RULETAKER_DATASET_ZIP_URL, RULETAKER_DATASET)

            self.dataset_name = dataset_name
            self.split_set = split_set
            self.dataset_path = (
                f"{RULETAKER_DATASET_FOLDER}/{self.dataset_name}/{self.split_set}.jsonl"
            )
            (
                self.contexts,
                self.questions,
                self.labels,
                self.depths,
            ) = self.__read_dataset("context", "questions", "text", "label")
        except DatasetNameError as err:
            print(err.message)
            print(f"The RuleTaker datasets are: {RULETAKER_SUB_DATASETS}")
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self,
        contexts_key: str,
        questions_key: str,
        questions_text_key: str,
        labels_key: str,
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        data = read_jsonl(self.dataset_path)
        contexts_list = []
        questions_list = []
        labels_list = []
        depths_list = []

        for i in data:
            for q in i[questions_key]:
                contexts_list.append(i[contexts_key])
                questions_list.append(q[questions_text_key])
                labels_list.append(RULETAKER_LABEL_TO_ID[str(q[str(labels_key)])])
                depths_list.append(q["meta"]["QDep"])

        return contexts_list, questions_list, labels_list, depths_list

    def __getitem__(self, index: int) -> Tuple[str, str, int, int]:
        return (
            self.contexts[index],
            self.questions[index],
            self.labels[index],
            self.depths[index],
        )

    def __str__(self) -> str:
        return f"The {self.split_set} set of {self.dataset_name}'s RuleTaker has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.contexts)
