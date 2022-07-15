from typing import List


class TaskError(Exception):
    """
    An error is raised when the task is wrong
    """

    def __init__(self, tasks: List[str]):
        self.message = (
            f"TaskError: Task is not found. Tasks that the model can are {tasks}\n"
        )


class LossError(Exception):
    """
    An error is raised when the loss is wrong
    """

    def __init__(self, losses: List[str]):
        self.message = f"LossError: Loss not suppported. Losses that the model can use are {losses}\n"
