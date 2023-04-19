import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from logitorch.datasets.utils import (
    DATASETS_FOLDER,
    DATASETS_ZIP_FOLDER,
    download_dataset,
    read_json,
    read_jsonl,
)


def test_download_dataset():
    test_url = "https://www.dropbox.com/s/5dpm2yjcm5h9hsd/proofwriter_dataset.zip?dl=1"
    test_dataset_name = "test_dataset"

    # Create temporary directories for test
    with TemporaryDirectory() as temp_dir:
        os.environ["CACHED_PATH_CACHE_ROOT"] = temp_dir
        download_dataset(test_url, test_dataset_name)

        assert os.path.exists(f"{DATASETS_FOLDER}/{test_dataset_name}")
        assert os.path.exists(f"{DATASETS_ZIP_FOLDER}/{test_dataset_name}.zip")


def test_read_jsonl():
    jsonl_content = [
        {"id": 1, "text": "Hello"},
        {"id": 2, "text": "World"},
    ]

    with TemporaryDirectory() as temp_dir:
        test_file_path = Path(temp_dir) / "test.jsonl"
        with open(test_file_path, "w", encoding="utf-8") as file:
            for item in jsonl_content:
                file.write(json.dumps(item) + "\n")

        result = read_jsonl(str(test_file_path))
        assert result == jsonl_content


def test_read_json():
    json_content = [
        {"id": 1, "text": "Hello"},
        {"id": 2, "text": "World"},
    ]

    with TemporaryDirectory() as temp_dir:
        test_file_path = Path(temp_dir) / "test.json"
        with open(test_file_path, "w", encoding="utf-8") as file:
            json.dump(json_content, file)

        result = read_json(str(test_file_path))
        assert result == json_content


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    shutil.rmtree(DATASETS_ZIP_FOLDER, ignore_errors=True)
    shutil.rmtree(DATASETS_FOLDER, ignore_errors=True)
    os.environ.pop("CACHED_PATH_CACHE_ROOT", None)
