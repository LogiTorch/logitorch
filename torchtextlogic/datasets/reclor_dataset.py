import os
from typing import List, Tuple, Union

from datasets.base import AbstractMCQADataset
from torchtextlogic.datasets.exceptions import SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_json,
)

RECLOR_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/4dabc3ea0cf6sre/reclor_dataset.zip?dl=1"
)
RECLOR_DATASET = "reclor_dataset"
RECLOR_DATASET_FOLDER = f"{DATASETS_FOLDER}/{RECLOR_DATASET}"


class ReClorDataset(AbstractMCQADataset):
    def __init__(self, split_set: str) -> None:
        super().__init__()
        try:
            if split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(RECLOR_DATASET_FOLDER):
                download_dataset(RECLOR_DATASET_ZIP_URL, RECLOR_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{RECLOR_DATASET_FOLDER}/{self.split_set}.json"
            (
                self.contexts,
                self.questions,
                self.answers,
                self.labels,
            ) = self.__read_dataset("context", "question", "answers", "label")
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self, contexts_key: str, questions_key: str, answers_key: str, labels_key: str
    ) -> Tuple[List[str], List[str], List[List[str]], List[int]]:
        data = read_json(self.dataset_path)
        contexts_list = []
        questions_list = []
        answers_list = []
        labels_list = []

        for i in data:
            tmp_answers = []
            for a in i[answers_key]:
                tmp_answers.append(a)
            contexts_list.append(i[contexts_key])
            questions_list.append(i[questions_key])
            answers_list.append(tmp_answers)

            if self.split_set == "train" or self.split_set == "val":
                labels_list.append(i[labels_key])

        return contexts_list, questions_list, answers_list, labels_list

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[str, str, List[str], int], Tuple[str, str, List[str]]]:

        if self.split_set == "test":
            return self.contexts[index], self.questions[index], self.answers[index]
        else:
            return (
                self.contexts[index],
                self.questions[index],
                self.answers[index],
                self.labels[index],
            )

    def __str__(self) -> str:
        return f"The {self.split_set} set of ReClor has {self.__len__()} instances"

    def __len__(self) -> int:
        return len(self.contexts)
