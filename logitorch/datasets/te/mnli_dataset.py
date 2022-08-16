import os
from typing import List, Tuple

import pandas as pd

from logitorch.datasets.base import AbstractTEDataset
from logitorch.datasets.exceptions import SplitSetError
from logitorch.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS_TRAIN_VAL,
    download_dataset,
)

MNLI_DATASET_ZIP_URL = "https://www.dropbox.com/s/t5mq9808mpvhvez/mnli_dataset.zip?dl=1"
MNLI_DATASET = "mnli_dataset"
MNLI_DATASET_FOLDER = f"{DATASETS_FOLDER}/{MNLI_DATASET}"
MNLI_LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
MNLI_ID_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


class MNLIDataset(AbstractTEDataset):
    def __init__(self, split_set: str) -> None:
        super().__init__()

        try:
            if split_set not in SPLIT_SETS_TRAIN_VAL:
                raise SplitSetError(SPLIT_SETS_TRAIN_VAL)
            if not os.path.exists(MNLI_DATASET_FOLDER):
                download_dataset(MNLI_DATASET_ZIP_URL, MNLI_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{MNLI_DATASET_FOLDER}/{self.split_set}.csv"
            self.premises, self.hypotheses, self.labels = self.__read_dataset(
                "premise", "hypothesis", "label"
            )
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self, premises_key: str, hypotheses_key: str, labels_key: str
    ) -> Tuple[List[str], List[str], List[int]]:
        data = pd.read_csv(self.dataset_path)
        premises_list = []
        hypotheses_list = []
        labels_list = []

        for premise, hypothesis, label in zip(
            data[premises_key].tolist(),
            data[hypotheses_key].tolist(),
            data[labels_key].tolist(),
        ):
            premises_list.append(str(premise))
            hypotheses_list.append(str(hypothesis))
            labels_list.append(label)

        return premises_list, hypotheses_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self.premises[index], self.hypotheses[index], self.labels[index]

    def __str__(self) -> str:
        return f"The {self.split_set} set of MNLI has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.premises)
