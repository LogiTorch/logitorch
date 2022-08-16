import json
import os
from pathlib import Path
from typing import Any, Dict, List
from zipfile import ZipFile

import requests
from tqdm import tqdm

from logitorch.datasets.exceptions import FileSizeError

CURRENT_PATH = str(Path(os.getenv("CACHED_PATH_CACHE_ROOT", Path.home() / ".cache")))
DATASETS_FOLDER = f"{CURRENT_PATH}/logitorch_datasets"
DATASETS_ZIP_FOLDER = f"{DATASETS_FOLDER}/tmp"

SPLIT_SETS = ["train", "val", "test"]
SPLIT_SETS_TRAIN_VAL = ["train", "val"]


def download_dataset(url: str, dataset_name: str) -> None:
    """Function to download datasets

    :param url: url of the dataset
    :type url: str
    :param dataset_name: dataset name
    :type dataset_name: str
    :raises FileSizeError: an error is raised if the dataset is not downloaded properly
    """
    if not os.path.exists(DATASETS_ZIP_FOLDER):
        os.makedirs(DATASETS_ZIP_FOLDER)

    dataset_zip_path = f"{DATASETS_ZIP_FOLDER}/{dataset_name}.zip"

    if dataset_zip_path not in os.listdir(DATASETS_ZIP_FOLDER):
        req = requests.get(url, stream=True)
        total_size = int(req.headers["content-length"])
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)

        with open(dataset_zip_path, "wb") as fw:
            for data in req.iter_content(block_size):
                t.update(len(data))
                fw.write(data)

        t.close()

    try:
        if total_size != 0 and t.n != total_size:
            raise FileSizeError()
        else:
            __extract_dataset_zip(dataset_zip_path, dataset_name)
    except FileSizeError as err:
        print(err.message)
        if os.path.exists(dataset_zip_path):
            os.remove(dataset_zip_path)


def read_jsonl(dataset_path: str) -> List[Dict[str, Any]]:
    """Function to read a JSONL file

    :param dataset_path: path of the dataset
    :type dataset_path: str
    :return: list of JSON objects
    :rtype: List[Dict[str, Any]]
    """
    with open(dataset_path, "r", encoding="utf-8") as out:
        jsonl = list(out)

    return [json.loads(i) for i in jsonl]


def read_json(dataset_path: str) -> List[Dict[str, Any]]:
    """Function to read a JSON file

    :param dataset_path: path of the dataset
    :type dataset_path: str
    :return: list of JSON objects
    :rtype: List[Dict[str, Any]]
    """
    with open(dataset_path, "r", encoding="utf-8") as out:
        json_file = json.loads(out.read())

    return json_file


def __extract_dataset_zip(dataset_zip_path: str, dataset_name: str) -> None:
    """Function to extract a dataset in zip extension

    :param dataset_zip_path: dataset in zip extension on disk
    :type dataset_zip_path: str
    :param dataset_name: dataset name
    :type dataset_name: str
    """
    dataset_path = f"{DATASETS_FOLDER}/{dataset_name}"
    with ZipFile(dataset_zip_path, "r") as zip_file:
        zip_file.extractall(dataset_path)
