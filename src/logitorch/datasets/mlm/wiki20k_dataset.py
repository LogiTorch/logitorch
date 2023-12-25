import os
from typing import List, Tuple

from logitorch.datasets.exceptions import DatasetNameError, SplitSetError
from logitorch.datasets.utils import DATASETS_FOLDER, download_dataset, read_jsonl

WIKI20K_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/yeh70n6etbg0a95/wiki20k_dataset.zip?dl=1"
)
WIKI20K_DATASET = "wiki20k_dataset"
WIKI20K_SUB_DATASETS = ["lm_wiki20k", "positive_lm_wiki20k", "negated_lm_wiki20k"]
WIKI20K_DATASET_FOLDER = f"{DATASETS_FOLDER}/{WIKI20K_DATASET}"


class Wiki20KDataset:
    """
    A class representing the Wiki20K dataset for RuleTaker.

    Attributes:
        dataset_name (str): The name of the dataset.
        dataset_path (str): The path to the dataset file.
        sentences (List[str]): The list of sentences in the dataset.
        labels (List[str]): The list of labels in the dataset.

    Methods:
        __init__(self, dataset_name: str, size: int = None) -> None:
            Initializes a Wiki20KDataset object.
        __read_dataset(self, sentences_key: str, labels_key: str, size: int = None) -> Tuple[List[str], List[str], List[int]]:
            Reads the dataset file and returns the sentences and labels.
        __getitem__(self, index: int) -> Tuple[str, str, int]:
            Returns the sentence, label, and index at the given index.
        __str__(self) -> str:
            Returns a string representation of the dataset.
        __len__(self) -> int:
            Returns the number of instances in the dataset.
    """

    def __init__(self, dataset_name: str, size: int = None) -> None:
        """
        Initializes a Wiki20KDataset object.

        Args:
            dataset_name (str): The name of the dataset.
            size (int, optional): The number of instances to load from the dataset. Defaults to None.
        """
        super().__init__()
        try:
            if dataset_name not in WIKI20K_SUB_DATASETS:
                raise DatasetNameError()
            if not os.path.exists(WIKI20K_DATASET_FOLDER):
                download_dataset(WIKI20K_DATASET_ZIP_URL, WIKI20K_DATASET)

            self.dataset_name = dataset_name
            self.dataset_path = f"{WIKI20K_DATASET_FOLDER}/{self.dataset_name}.jsonl"
            self.sentences, self.labels = self.__read_dataset("sentence", "label", size)
        except DatasetNameError as err:
            print(err.message)
            print(f"The RuleTaker datasets are: {WIKI20K_SUB_DATASETS}")
        except SplitSetError as err:
            print(err.message)

    def __read_dataset(
        self, sentences_key: str, labels_key: str, size: int = None
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Reads the Wiki20K dataset.

        Args:
            sentences_key (str): The key for the sentences in the dataset file.
            labels_key (str): The key for the labels in the dataset file.
            size (int, optional): The number of instances to read from the dataset. Defaults to None.

        Returns:
            Tuple[List[str], List[str], List[int]]: A tuple containing the sentences, labels, and indices.
        """
        data = read_jsonl(self.dataset_path)
        sentences_list = []
        labels_list = []

        if size is None:
            size = len(data)

        for i in data[:size]:
            sentences_list.append(i[sentences_key])
            labels_list.append(i[labels_key])

        return sentences_list, labels_list

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        """
        Returns the sentence, label, and index at the given index.

        Args:
            index (int): The index of the instance to retrieve.

        Returns:
            Tuple[str, str, int]: A tuple containing the sentence, label, and index.
        """
        return self.sentences[index], self.labels[index]

    def __str__(self) -> str:
        """
        Returns a string representation of the dataset.

        Returns:
            str: A string representation of the dataset.
        """
        return (
            f"The set of {self.dataset_name}'s RuleTaker has {self.__len__()} instances"
        )

    def __len__(self) -> int:
        """
        Returns the number of instances in the dataset.

        Returns:
            int: The number of instances in the dataset.
        """
        return len(self.sentences)
