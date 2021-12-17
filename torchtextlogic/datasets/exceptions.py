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
