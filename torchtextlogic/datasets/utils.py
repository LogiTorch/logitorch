import os
from zipfile import ZipFile

import requests
from tqdm import tqdm

CURRENT_PATH = os.getcwd()
DATASET_FOLDER_NAME = f"{CURRENT_PATH}dataset/"
DATASET_ZIP_FOLDER_NAME = f"{DATASET_FOLDER_NAME}tmp/"


class DatasetNameError(Exception):
    """
    An error is raised when the dataset name is wrong
    """

    def __init__(self):
        self.message = "DatasetNameError: Dataset name is wrong"


class FileSizeError(Exception):
    """
    An error is raised when the downloaded dataset has a wrong size
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
    if not os.path.exists(DATASET_FOLDER_NAME):
        os.mkdir(DATASET_FOLDER_NAME)

    dataset_zip_name_on_disk = f"{DATASET_ZIP_FOLDER_NAME}{dataset_name}"

    if dataset_zip_name_on_disk not in os.listdir(DATASET_ZIP_FOLDER_NAME):
        req = requests.get(url, stream=True)
        total_size = int(req.headers["content-length"])
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)

        with open(dataset_zip_name_on_disk, "wb") as fw:
            for data in req.iter_content(block_size):
                t.update(len(data))
                fw.write(data)

        t.close()

    try:
        if total_size != 0 and t.n != total_size:
            raise FileSizeError()
        else:
            __extract_dataset(dataset_zip_name_on_disk, dataset_name)
    except FileExistsError as err:
        print(err.message)

        if os.path.exists(dataset_zip_name_on_disk):
            os.remove(dataset_zip_name_on_disk)


def __extract_dataset(dataset_zip_name_on_disk: str, dataset_name: str) -> None:
    """Function to extract a dataset in zip extension

    :param dataset_zip_name_on_disk: dataset in zip extension on disk
    :type dataset_zip_name_on_disk: str
    :param dataset_name: dataset name
    :type dataset_name: str
    """
    dataset_name_on_disk = f"{DATASET_FOLDER_NAME}{dataset_name}"
    with ZipFile(dataset_zip_name_on_disk, "r") as zip_file:
        zip_file.extractall(dataset_name_on_disk)
