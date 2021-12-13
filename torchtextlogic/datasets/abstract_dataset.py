from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch.utils.data import Dataset


class AbstractQADataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def read_dataset(self, dataset_name: str) -> None:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, str, Any]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class AbstracTEDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def read_dataset(self, dataset_name: str) -> None:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
