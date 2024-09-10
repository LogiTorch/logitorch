import os
from typing import List, Tuple, Union

from logitorch.datasets.base import AbstractMCQADataset
from logitorch.datasets.exceptions import SplitSetError
from logitorch.datasets.utils import SPLIT_SETS

from datasets import load_dataset

LOGIQA2_DATASET = "jeggers/logiqa2_formatted"


class LogiQA2Dataset(AbstractMCQADataset):
    """
    A dataset class for LogiQA2, a Multiple Choice Question Answering dataset.
    """

    def __init__(self, split_set: str) -> None:
        """
        Initialize the LogiQA2Dataset.

        Args:
            split_set (str): The split set of the dataset (train, validation, or test).

        Raises:
            SplitSetError: If an invalid split set is provided.
        """
        super().__init__()
        try:
            if split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            self.split_set = split_set
            if split_set == "val":
                split_set = "validation"

            self.dataset = load_dataset(LOGIQA2_DATASET, split=split_set)
            print(self.dataset)
            self.contexts, self.questions, self.answers, self.labels = (
                self.__process_dataset()
            )
        except SplitSetError as err:
            print(err.message)

    def __process_dataset(
        self,
    ) -> Tuple[List[str], List[str], List[List[str]], List[int]]:
        """
        Process the LogiQA2 dataset.

        Returns:
            Tuple[List[str], List[str], List[List[str]], List[int]]: A tuple containing the contexts, questions,
            answers, and labels of the dataset.
        """
        contexts = self.dataset["text"]
        questions = self.dataset["question"]
        answers = [item["formatted_options"] for item in self.dataset]
        labels = [ord(item["answer_char"]) - ord("A") for item in self.dataset]

        return contexts, questions, answers, labels

    def __getitem__(self, index: int) -> Tuple[str, str, List[str], int]:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            Tuple[str, str, List[str], int]: A tuple containing the context,
            question, answers, and label of the item.
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
            str: A string representation of the dataset.
        """
        return f"The {self.split_set} set of LogiQA2 has {self.__len__()} instances"

    def __len__(self) -> int:
        """
        Returns the number of instances in the dataset.

        Returns:
            int: The number of instances in the dataset.
        """
        return len(self.contexts)
