import os
from zipfile import ZipFile

import requests

from tqdm import tqdm

DATASET_FOLDER_NAME = "dataset/"
class EmbeddingFileNameError(Exception):
    """
    Raised when a pretrained word embedding file name is wrong
    """

    def __init__(self):
        self.message = (
            "EmbeddingFileNameError: Pretrained word embedding file name is wrong"
        )


class FileSizeError(Exception):
    """
    Raised when a downloaded file has a wrong size
    """

    def __init__(self):
        self.message = "FileSizeError: Wrong file size"


def download_dataset(url: str, src_dataset: str) -> None:
    pass


def extract_dataset(src_dataset: str) -> None:
    pass
