from typing import Any, Tuple

from torchtextlogic.datasets.abstract_dataset import AbstractQADataset


class RuleTakerDataset(AbstractQADataset):
    def __init__(self) -> None:
        super().__init__()

    def read_dataset(self, dataset_name: str) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        pass

    def __str__(self) -> str:
        pass

    def __len__(self) -> int:
        pass