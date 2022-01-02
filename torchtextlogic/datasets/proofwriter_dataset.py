import os
from typing import List, Tuple

from torchtextlogic.datasets.exceptions import DatasetNameError, SplitSetError
from torchtextlogic.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

PROOFWRITER_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/5dpm2yjcm5h9hsd/proofwriter_dataset.zip?dl=1"
)
PROOFWRITER_DATASET = "proofwriter_dataset"
PROOFWRITER_SUB_DATASETS = [
    "birds-electricity",
    "depth-0",
    "depth-1",
    "depth-2",
    "depth-3",
    "depth-3ext",
    "depth-3ext-NatLang",
    "depth-5",
    "NatLang",
]
PROOFWRITER_DATASET_FOLDER = f"{DATASETS_FOLDER}/{PROOFWRITER_DATASET}"
PROOFWRITER_LABEL_TO_ID = {False: 0, True: 1}
PROOFWRITER_ID_TO_LABEL = {0: "False", 1: "True"}


class ProofWriterDataset:
    pass
