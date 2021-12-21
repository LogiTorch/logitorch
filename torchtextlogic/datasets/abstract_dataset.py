from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from torch.utils.data import Dataset


class AbstractDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class AbstractMCQADataset(AbstractDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, List[str, str, str, str], Any]:
        pass


class AbstractTEDataset(AbstractDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        pass


class AbstractQADataset(AbstractDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        pass
