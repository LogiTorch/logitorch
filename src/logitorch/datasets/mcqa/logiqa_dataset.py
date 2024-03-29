import os
from typing import List, Tuple

from logitorch.datasets.base import AbstractMCQADataset
from logitorch.datasets.exceptions import SplitSetError
from logitorch.datasets.utils import DATASETS_FOLDER, SPLIT_SETS, download_dataset

LOGIQA_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/7q0eaosyd5zu5st/logiqa_dataset.zip?dl=1"
)
LOGIQA_DATASET = "logiqa_dataset"
LOGIQA_DATASET_FOLDER = f"{DATASETS_FOLDER}/{LOGIQA_DATASET}"
LOGIQA_LABEL_TO_ID = {"a": 0, "b": 1, "c": 2, "d": 3}
LOGIQA_ID_TO_LABEL = {0: "a", 1: "b", 2: "c", 3: "d"}


class LogiQADataset(AbstractMCQADataset):
    """
    A dataset class for LogiQA, a multiple-choice question answering dataset.
    """

    def __init__(self, split_set: str) -> None:
        """
        Initialize the LogiQADataset.

        Args:
            split_set (str): The split set of the dataset to use.

        Raises:
            SplitSetError: If the split set is not valid.
        """
        super().__init__()
        try:
            if split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(LOGIQA_DATASET_FOLDER):
                download_dataset(LOGIQA_DATASET_ZIP_URL, LOGIQA_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{LOGIQA_DATASET_FOLDER}/{split_set}.txt"
            (
                self.contexts,
                self.questions,
                self.answers,
                self.labels,
            ) = self.__read_dataset()
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(self) -> Tuple[List[str], List[str], List[List[str]], List[int]]:
        """
        Read the LogiQA dataset.

        Returns:
            Tuple[List[str], List[str], List[List[str]], List[int]]: The dataset as a tuple of lists.
        """
        with open(self.dataset_path, "r", encoding="utf-8") as out:
            data = out.read().split("\n\n")
        contexts_list = []
        questions_list = []
        answers_list = []
        labels_list = []

        for i in data:
            tmp_splited_i = i.split("\n")
            contexts_list.append(tmp_splited_i[1])
            questions_list.append(tmp_splited_i[2])
            answers_list.append(
                [tmp_splited_i[3], tmp_splited_i[4], tmp_splited_i[5], tmp_splited_i[6]]
            )
            labels_list.append(LOGIQA_LABEL_TO_ID[tmp_splited_i[0]])

        return contexts_list, questions_list, answers_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, List[str], int]:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[str, str, List[str], int]: The item as a tuple of context, question, answers, and label.
        """
        return (
            self.contexts[index],
            self.questions[index],
            self.answers[index],
            self.labels[index],
        )

    def __str__(self) -> str:
        """
        Get a string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return f"The {self.split_set} set of LogiQA has {self.__len__()} instances"

    def __len__(self) -> int:
        """
        Returns the number of instances in the dataset.

        Returns:
            int: The number of instances in the dataset.
        """
        return len(self.contexts)
