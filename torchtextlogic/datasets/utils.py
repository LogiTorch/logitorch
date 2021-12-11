import os
from zipfile import ZipFile

import requests
from tqdm import tqdm

DATASET_FOLDER_NAME = "dataset/"
DATASET_ZIP_FOLDER_NAME = f"{DATASET_FOLDER_NAME}tmp"


class DatasetNameError(Exception):
    """
    Raised when the dataset name is wrong
    """

    def __init__(self):
        self.message = "DatasetNameError: Dataset name is wrong"


class FileSizeError(Exception):
    """
    Raised when the downloaded dataset has a wrong size
    """

    def __init__(self):
        self.message = "FileSizeError: Wrong file size"


def download_dataset(url: str, dataset_name: str) -> None:
    """Function to download datasets

    :param url: url of the dataset
    :type url: str
    :param dataset_name: dataset name
    :type dataset_name: str
    :raises FileSizeError: an error is raised if the dataset is not downloaded properly
    """
    dataset_name_on_disk = f"{DATASET_ZIP_FOLDER_NAME}{dataset_name}"

    if dataset_name_on_disk not in os.listdir(DATASET_ZIP_FOLDER_NAME):
        req = requests.get(url, stream=True)
        total_size = int(req.headers["content-length"])
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)

        with open(dataset_name_on_disk, "wb") as fw:
            for data in req.iter_content(block_size):
                t.update(len(data))
                fw.write(data)

        t.close()

    try:
        if total_size != 0 and t.n != total_size:
            raise FileSizeError()
    except FileExistsError as err:
        print(err.mesage)

        if os.path.exists(dataset_name_on_disk):
            os.remove(dataset_name_on_disk)


def extract_dataset(dataset_name: str) -> None:
    pass
