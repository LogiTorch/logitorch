from typing import Any, Tuple


class ModelNotCompatibleError(Exception):
    """
    An error is raised when the model is not compatible with the dataset.
    """

    def __init__(self, compatible_models: Tuple[Any]):
        self.message = f"ModelNotCompatibleError: Model is not compatible with the dataset.\nThe compatible list of models are {compatible_models}"
