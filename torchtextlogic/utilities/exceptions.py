class OutOfRangeError(Exception):
    """
    An error is raised when the value is out of the range
    """

    def __init__(self, min_value: float, max_value: float):
        self.message = (
            f"OutOfRangeError: The value must be betwen [{min_value}, {max_value}]"
        )
