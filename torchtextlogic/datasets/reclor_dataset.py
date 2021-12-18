from typing import Any, Tuple

from torchtextlogic.datasets.abstract_dataset import AbstractMCQADataset


class ReClorDataset(AbstractMCQADataset):
    def __init__(self) -> None:
        super().__init__()

    def __read_dataset(self) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[str, str, str, Any]:
        pass

    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        pass
