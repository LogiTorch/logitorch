import os
from typing import List, Tuple

from logitorch.datasets.base import AbstractQADataset
from logitorch.datasets.exceptions import SplitSetError
from logitorch.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS_TRAIN_VAL,
    download_dataset,
    read_jsonl,
)

FOLIO_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/ymtfzkhulzyiup1/folio_dataset.zip?dl=1"
)
FOLIO_DATASET = "folio_dataset"
FOLIO_DATASET_FOLDER = f"{DATASETS_FOLDER}/{FOLIO_DATASET}"
FOLIO_LABEL_TO_ID = {"False": 0, "True": 1, "Uncertain": 2, "Unknown": 2}
FOLIO_ID_TO_LABEL = {0: "False", 1: "True", 2: "Uncertain/Unknown"}


class FOLIODataset(AbstractQADataset):
    def __init__(self, split_set: str) -> None:
        super().__init__()
        try:
            if split_set not in SPLIT_SETS_TRAIN_VAL:
                raise SplitSetError(SPLIT_SETS_TRAIN_VAL)

            if not os.path.exists(FOLIO_DATASET_FOLDER):
                download_dataset(FOLIO_DATASET_ZIP_URL, FOLIO_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{FOLIO_DATASET_FOLDER}/{self.split_set}.jsonl"
            (
                self.contexts,
                self.questions,
                self.labels,
                self.contexts_fol,
            ) = self.__read_dataset("premises", "conclusion", "label", "premises-FOL")
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self,
        contexts_key: str,
        questions_key: str,
        labels_key: str,
        contexts_fol_key: str,
    ) -> Tuple[List[List[str]], List[str], List[int], List[List[str]]]:
        data = read_jsonl(self.dataset_path)
        contexts_list = []
        questions_list = []
        labels_list = []
        contexts_fol_list = []

        for i in data:
            contexts_list.append(i[contexts_key])
            questions_list.append(i[questions_key])
            labels_list.append(FOLIO_LABEL_TO_ID[str(i[str(labels_key)])])
            contexts_fol_list.append(i[contexts_fol_key])

        return contexts_list, questions_list, labels_list, contexts_fol_list

    def __getitem__(self, index: int) -> Tuple[List[str], str, int, List[str]]:
        return (
            self.contexts[index],
            self.questions[index],
            self.labels[index],
            self.contexts_fol[index],
        )

    def __str__(self) -> str:
        return f"The {self.split_set} set of FOLIO has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.contexts)
