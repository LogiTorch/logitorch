import os
from typing import List, Tuple

from logitorch.datasets.base import AbstractMCQADataset
from logitorch.datasets.exceptions import SplitSetError
from logitorch.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_json,
)

ARLSAT_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/yuaoz1kon66w2o6/arlsat_dataset.zip?dl=1"
)
ARLSAT_DATASET = "arlsat_dataset"
ARLSAT_DATASET_FOLDER = f"{DATASETS_FOLDER}/{ARLSAT_DATASET}"
ARLSAT_LABEL_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
ARLSAT_ID_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


class ARLSATDataset(AbstractMCQADataset):
    def __init__(self, split_set: str) -> None:
        """
        The constructor takes in a string as an argument and checks if it is in the list of split sets. If
        it is not, it raises an error. If it is, it checks if the dataset folder exists. If it does not,
        it downloads the dataset. It then sets the split set to the argument passed in, sets the dataset
        path to the dataset folder and the split set, and reads the dataset

        :param split_set: The split set to use
        :type split_set: str
        """
        super().__init__()
        try:
            if split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(ARLSAT_DATASET_FOLDER):
                download_dataset(ARLSAT_DATASET_ZIP_URL, ARLSAT_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{ARLSAT_DATASET_FOLDER}/{self.split_set}.json"
            (
                self.contexts,
                self.questions,
                self.answers,
                self.labels,
            ) = self.__read_dataset(
                "passage", "questions", "question", "options", "answer"
            )
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self,
        contexts_key: str,
        questions_key: str,
        questions_text_key: str,
        answers_key: str,
        labels_key: str,
    ) -> Tuple[List[str], List[str], List[List[str]], List[int]]:
        """
        This function reads a json file and returns a tuple of lists

        :param contexts_key: str,
        :type contexts_key: str
        :param questions_key: str,
        :type questions_key: str
        :param questions_text_key: str = "question"
        :type questions_text_key: str
        :param answers_key: str = "answers"
        :type answers_key: str
        :param labels_key: str,
        :type labels_key: str
        :return: A tuple of lists.
        """
        data = read_json(self.dataset_path)
        contexts_list = []
        questions_list = []
        answers_list = []
        labels_list = []

        for i in data:
            for q in i[questions_key]:
                tmp_answers = []
                for a in q[answers_key]:
                    tmp_answers.append(a)
                contexts_list.append(i[contexts_key])
                questions_list.append(q[questions_text_key])
                answers_list.append(tmp_answers)
                labels_list.append(ARLSAT_LABEL_TO_ID[q[labels_key]])

        return contexts_list, questions_list, answers_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, List[str], int]:
        """
        This function takes in an index and returns a tuple of the context, question, answer, and label at
        that index

        :param index: The index of the data point in the dataset
        :type index: int
        :return: A tuple of the context, question, answer, and label.
        """
        return (
            self.contexts[index],
            self.questions[index],
            self.answers[index],
            self.labels[index],
        )

    def __str__(self) -> str:
        """
        This function returns a string that says "The [split_set] set of ARLSAT has [number of instances]
        instances"

        The function takes in the split_set as an argument
        :return: The length of the dataset
        """
        return f"The {self.split_set} set of ARLSAT has {self.__len__()} instances"

    def __len__(self) -> int:
        """
        This function returns the length of the contexts list
        :return: The length of the contexts list.
        """
        return len(self.contexts)
