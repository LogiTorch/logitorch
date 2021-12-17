import json
import os
from typing import Any, Dict, List
from zipfile import ZipFile

import requests
from torchtextlogic.datasets.exceptions import FileSizeError
from tqdm import tqdm

CURRENT_PATH = os.getcwd()
DATASETS_FOLDER_NAME = f"{CURRENT_PATH}/torchtextlogic_datasets"
DATASETS_ZIP_FOLDER_NAME = f"{DATASETS_FOLDER_NAME}/tmp"


def download_dataset(url: str, dataset_name: str) -> None:
    """Function to download datasets

    :param url: url of the dataset
    :type url: str
    :param dataset_name: dataset name
    :type dataset_name: str
    :raises FileSizeError: an error is raised if the dataset is not downloaded properly
    """
    if not os.path.exists(DATASETS_ZIP_FOLDER_NAME):
        os.makedirs(DATASETS_ZIP_FOLDER_NAME)

    dataset_zip_name_on_disk = f"{DATASETS_ZIP_FOLDER_NAME}/{dataset_name}.zip"

    if dataset_zip_name_on_disk not in os.listdir(DATASETS_ZIP_FOLDER_NAME):
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
            __extract_dataset_zip(dataset_zip_name_on_disk, dataset_name)
    except FileSizeError as err:
        print(err.message)
        if os.path.exists(dataset_zip_name_on_disk):
            os.remove(dataset_zip_name_on_disk)


def read_jsonl(dataset_src: str) -> List[Dict[str, Any]]:
    """Function to read JSONL file

    :param dataset_src: path of the dataset
    :type dataset_src: str
    :return: list of JSON objects
    :rtype: List[Dict[str, Any]]
    """
    with open(dataset_src, "r", encoding="utf-8") as out:
        jsonl = list(out)

    return [json.loads(i) for i in jsonl]


def __extract_dataset_zip(dataset_zip_name_on_disk: str, dataset_name: str) -> None:
    """Function to extract a dataset in zip extension

    :param dataset_zip_name_on_disk: dataset in zip extension on disk
    :type dataset_zip_name_on_disk: str
    :param dataset_name: dataset name
    :type dataset_name: str
    """
    dataset_name_on_disk = f"{DATASETS_FOLDER_NAME}/{dataset_name}"
    with ZipFile(dataset_zip_name_on_disk, "r") as zip_file:
        zip_file.extractall(dataset_name_on_disk)
