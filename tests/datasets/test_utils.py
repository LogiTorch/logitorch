import os

from torchtextlogic.datasets.utils import DATASETS_FOLDER_NAME, download_dataset


def test_download_dataset() -> None:
    url = "https://www.dropbox.com/s/znrr7a056zpxcxh/proofwriter_dataset.zip?dl=1"
    dataset_name = "proofwriter_dataset"
    dataset_name_on_disk = f"{DATASETS_FOLDER_NAME}{dataset_name}"
    download_dataset(url, dataset_name)
    if os.path.exists(dataset_name_on_disk):
        assert True
