class ModelNotCompatibleError(Exception):
    """
    An error is raised when the model is not compatible with the dataset.
    """

    def __init__(self):
        self.message = (
            "ModelNotCompatibleError: Model is not compatible with the dataset."
        )
