import os
from typing import List, Tuple, Union

from logitorch.datasets.base import AbstractMCQADataset
from logitorch.datasets.exceptions import SplitSetError
from logitorch.datasets.utils import (
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
        """
        The constructor takes in a string as an argument and checks if it is in the list of split sets

        :param split_set: The split set to use
        :type split_set: str
        """
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
        """
        This function reads a json file and returns a tuple of 4 lists: contexts, questions, answers, and labels

        :param cnotexts_key: The key in the JSON file that contains the context
        :type contexts_key: str
        :param questions_key: The key in the JSON file that contains the questions
        :type questions_key: str
        :param answers_key: The key in the JSON file that contains the answers
        :type answers_key: str
        :param labels_key: The key in the JSON file that contains the labels
        :type labels_key: str
        :return: A tuple of lists.
        """
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
        """
        This function returns a tuple of the context, question, and answer for the test set, and a tuple
        of the context, question, answer, and label for the train and dev sets

        :param index: The index of the data point we want to access
        :type index: int
        :return: A tuple of the context, question, answer, and label
        """
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
        """
        This function returns a string that contains the name of the split set and the number of
        instances in the split set

        :return: The number of instances in the dataset.
        """
        return f"The {self.split_set} set of ReClor has {self.__len__()} instances"

    def __len__(self) -> int:
        """
        This function returns the length of the contexts list

        :return: The length of the contexts list.
        """
        return len(self.contexts)
