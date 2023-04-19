import os
import shutil
from tempfile import TemporaryDirectory

import pytest

from logitorch.datasets.te.mnli_dataset import (
    MNLI_DATASET,
    MNLI_DATASET_FOLDER,
    MNLI_DATASET_ZIP_URL,
    MNLIDataset,
    download_dataset,
)


@pytest.fixture(scope="module")
def mock_data_dir():
    with TemporaryDirectory() as temp_dir:
        os.environ["CACHED_PATH_CACHE_ROOT"] = temp_dir

        # Download the dataset
        download_dataset(MNLI_DATASET_ZIP_URL, MNLI_DATASET)

        yield temp_dir

        shutil.rmtree(MNLI_DATASET_FOLDER, ignore_errors=True)
        os.environ.pop("CACHED_PATH_CACHE_ROOT", None)


def test_mnli_dataset_length(mock_data_dir):
    train_set = MNLIDataset("train")
    assert len(train_set) > 0

    val_set = MNLIDataset("val")
    assert len(val_set) > 0


def test_mnli_dataset_content(mock_data_dir):
    train_set = MNLIDataset("train")

    for premise, hypothesis, label in train_set:
        assert isinstance(premise, str)
        assert isinstance(hypothesis, str)
        assert isinstance(label, int)
        assert 0 <= label <= 2
