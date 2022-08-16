import os
from typing import List, Tuple

from logitorch.datasets.base import AbstractTEDataset
from logitorch.datasets.exceptions import SplitSetError
from logitorch.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

CONTROL_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/rmcqituydxacuhv/control_dataset.zip?dl=1"
)
CONTROL_DATASET = "control_dataset"
CONTROL_DATASET_FOLDER = f"{DATASETS_FOLDER}/{CONTROL_DATASET}"
CONTROL_LABEL_TO_ID = {"c": 0, "e": 1, "n": 2}
CONTROL_ID_TO_LABEL = {0: "c", 1: "e", 2: "n"}


class ControlDataset(AbstractTEDataset):
    def __init__(self, split_set: str) -> None:
        """
        The function takes in a string as an argument and returns None.

        The function first checks if the string is in the list of split sets. If it is not, it raises an
        error.

        If the string is in the list of split sets, the function checks if the control dataset folder
        exists. If it does not, it downloads the dataset.

        The function then sets the split set to the string that was passed in as an argument.

        The function then sets the dataset path to the control dataset folder and the split set.

        The function then reads the dataset and sets the premises, hypotheses, and labels to the read
        dataset.

        If there is an error, the function prints the error message.

        :param split_set: str
        :type split_set: str
        """
        super().__init__()

        try:
            if split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(CONTROL_DATASET_FOLDER):
                download_dataset(CONTROL_DATASET_ZIP_URL, CONTROL_DATASET)

            self.split_set = split_set
            self.dataset_path = f"{CONTROL_DATASET_FOLDER}/{self.split_set}.jsonl"
            self.premises, self.hypotheses, self.labels = self.__read_dataset(
                "premise", "hypothesis", "label"
            )
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self, premises_key: str, hypotheses_key: str, labels_key: str
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        It reads a jsonl file and returns a tuple of lists of strings and integers

        :param premises_key: str, hypotheses_key: str, labels_key: str
        :type premises_key: str
        :param hypotheses_key: str = "hypothesis"
        :type hypotheses_key: str
        :param labels_key: str = "label"
        :type labels_key: str
        :return: A tuple of lists of strings and integers.
        """
        data = read_jsonl(self.dataset_path)
        premises_list = []
        hypotheses_list = []
        labels_list = []

        for i in data:
            premises_list.append(str(i[premises_key]))
            hypotheses_list.append(str(i[hypotheses_key]))
            labels_list.append(CONTROL_LABEL_TO_ID[str(i[labels_key])])

        return premises_list, hypotheses_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        """
        This function returns the premise, hypothesis, and label of the index of the dataset

        :param index: The index of the data point you want to access
        :type index: int
        :return: The premise, hypothesis, and label for the given index.
        """
        return self.premises[index], self.hypotheses[index], self.labels[index]

    def __str__(self) -> str:
        """
        This function returns a string that describes the split set of ConTRoL and the number of
        instances in the split set
        :return: The number of instances in the split set.
        """
        return f"The {self.split_set} set of ConTRoL has {self.__len__()} instances"

    def __len__(self) -> int:
        """
        This function returns the length of the premises list
        :return: The length of the premises.
        """
        return len(self.premises)
