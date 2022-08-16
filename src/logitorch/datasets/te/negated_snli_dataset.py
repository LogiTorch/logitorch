import os
from typing import List, Tuple

import pandas as pd

from logitorch.datasets.base import AbstractTEDataset
from logitorch.datasets.utils import DATASETS_FOLDER, download_dataset

NEGATED_SNLI_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/dkdiagvxtzrmlxm/negated_snli_dataset.zip?dl=1"
)
NEGATED_SNLI_DATASET = "negated_snli_dataset"
NEGATED_SNLI_DATASET_FOLDER = f"{DATASETS_FOLDER}/{NEGATED_SNLI_DATASET}"
NEGATED_SNLI_LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
NEGATED_SNLI_ID_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


class NegatedSNLIDataset(AbstractTEDataset):
    def __init__(self) -> None:
        super().__init__()

        if not os.path.exists(NEGATED_SNLI_DATASET_FOLDER):
            download_dataset(NEGATED_SNLI_DATASET_ZIP_URL, NEGATED_SNLI_DATASET)

        self.dataset_path = f"{NEGATED_SNLI_DATASET_FOLDER}/test.txt"
        self.premises, self.hypotheses, self.labels = self.__read_dataset(
            "Text", "Hypothesis", "gold_label"
        )

    def __read_dataset(
        self, premises_key: str, hypotheses_key: str, labels_key: str
    ) -> Tuple[List[str], List[str], List[int]]:
        data = pd.read_csv(self.dataset_path, sep="\t", encoding="cp1252")
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
            labels_list.append(NEGATED_SNLI_LABEL_TO_ID[str(label)])

        return premises_list, hypotheses_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        return self.premises[index], self.hypotheses[index], self.labels[index]

    def __str__(self) -> str:
        return f"The test set of the Negated SNLI has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.premises)
