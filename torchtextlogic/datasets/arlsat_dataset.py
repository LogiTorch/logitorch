import os
from typing import List, Tuple

from torchtextlogic.datasets.abstract_dataset import AbstractMCQADataset
from torchtextlogic.datasets.exceptions import SplitSetError
from torchtextlogic.datasets.utils import (
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
    """[summary]"""

    def __init__(self, split_set: str) -> None:
        """[summary]

        :param split_set: [description]
        :type split_set: str
        :raises SplitSetError: [description]
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
        """[summary]

        :param contexts_key: [description]
        :type contexts_key: str
        :param questions_key: [description]
        :type questions_key: str
        :param questions_text_key: [description]
        :type questions_text_key: str
        :param answers_key: [description]
        :type answers_key: str
        :param labels_key: [description]
        :type labels_key: str
        :return: [description]
        :rtype: Tuple[List[str], List[str], List[List[str]], List[int]]
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
        """[summary]

        :param index: [description]
        :type index: int
        :return: [description]
        :rtype: Tuple[str, str, List[str], Any]
        """
        return (
            self.contexts[index],
            self.questions[index],
            self.answers[index],
            self.labels[index],
        )

    def __str__(self) -> str:
        """[summary]

        :return: [description]
        :rtype: str
        """
        return f"The {self.split_set} set of ARLSAT has {self.__len__()} instances"

    def __len__(self) -> int:
        """[summary]

        :return: [description]
        :rtype: int
        """
        return len(self.contexts)
