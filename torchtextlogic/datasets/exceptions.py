from typing import List


class DatasetNameError(Exception):
    """
    An error is raised when the dataset name is wrong
    """

    def __init__(self):
        self.message = "DatasetNameError: Dataset name is wrong"


class SplitSetError(Exception):
    """
    An error is raised when the split set is wrong
    """

    def __init__(self, split_sets: List[str]):
        self.message = (
            f"SplitSetError: Dataset name is wrong\nThe split sets are: {split_sets}"
        )


class FileSizeError(Exception):
    """
    An error is raised when the downloaded dataset has a wrong size
    """

    def __init__(self):
        self.message = "FileSizeError: Wrong file size"
