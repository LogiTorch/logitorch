from typing import List


class TaskError(Exception):
    """
    An error is raised when the task is wrong
    """

    def __init__(self, tasks: List[str]):
        self.message = (
            f"TaskError: Task is not found. Tasks that the model can are {tasks}\n"
        )
