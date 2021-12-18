import os
from typing import Any, Dict, List, Tuple

from torch.utils import data
from torchtextlogic.datasets.abstract_dataset import AbstractQADataset
from torchtextlogic.datasets.exceptions import DatasetNameError, SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER,
    download_dataset,
    read_jsonl,
)

RULETAKER_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/4j6jcc7ld5rf2tf/ruletaker_dataset.zip?dl=1"
)
RULETAKER_DATASET = "ruletaker_dataset"
RULETAKER_SUB_DATASETS = [
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
SPLIT_SETS = ["train", "dev", "test"]
RULETAKER_DATASET_FOLDER = f"{DATASETS_FOLDER}/{RULETAKER_DATASET}/ruletaker"


class RuleTakerDataset(AbstractQADataset):
    def __init__(self, dataset_name: str, split_set: str) -> None:
        super().__init__()
        try:
            if dataset_name not in RULETAKER_SUB_DATASETS:
                raise DatasetNameError()
            if split_set not in SPLIT_SETS:
                raise SplitSetError()
        except DatasetNameError as err:
            print(err.message)
            print(f"The RuleTaker datasets are: {RULETAKER_SUB_DATASETS}")
        except SplitSetError as err:
            print(err.message)
            print(f"The split sets are: {SPLIT_SETS}")

        if not os.path.exists(RULETAKER_DATASET_FOLDER):
            download_dataset(RULETAKER_DATASET_ZIP_URL, RULETAKER_DATASET)

        self.dataset_name = dataset_name
        self.split_set = split_set
        self.dataset_path = (
            f"{RULETAKER_DATASET_FOLDER}/{self.dataset_name}/{self.split_set}.jsonl"
        )
        self.contexts, self.questions, self.labels = self.__read_dataset(
            "context", "questions", "text", "label"
        )

    def __read_dataset(
        self, contexts_key: str, questions_key: str, texts_key: str, labels_key: str
    ) -> Tuple[List[str], List[str], List[str]]:
        data = read_jsonl(self.dataset_path)
        contexts_list = []
        questions_list = []
        labels_list = []

        for i in data:
            for q in i[questions_key]:
                contexts_list.append(i[contexts_key])
                questions_list.append(q[texts_key])
                labels_list.append(q[labels_key])

        return contexts_list, questions_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        return self.contexts[index], self.questions[index], self.labels[index]

    def __str__(self) -> str:
        return f"The {self.split_set} set of {self.dataset_name}'s RuleTaker has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.contexts)
