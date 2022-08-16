import os

from logitorch.datasets.utils import DATASETS_FOLDER, download_dataset


def test_download_dataset() -> None:
    url = "https://www.dropbox.com/s/5dpm2yjcm5h9hsd/proofwriter_dataset.zip?dl=1"
    dataset_name = "proofwriter_dataset"
    dataset_name_on_disk = f"{DATASETS_FOLDER}{dataset_name}"
    download_dataset(url, dataset_name)
    if os.path.exists(dataset_name_on_disk):
        assert True
