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


class TaskError(Exception):
    """
    An error is raised when the task is wrong
    """

    def __init__(self):
        self.message = "TaskError: Task is not found\n"


class AbductionClosedWorldAssumptionError(Exception):
    """
    An error is raised when the abduction task is chosen in a closed-world assumption setting
    """

    def __init__(self):
        self.message = "AbductionClosedWolrdAssumptionError: Abduction task exists only in open-world assumption (OWA) setting"
