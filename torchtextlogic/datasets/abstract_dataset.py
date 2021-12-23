from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from torch.utils.data import Dataset


class AbstractDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class AbstractMCQADataset(AbstractDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, List[str], Optional[Any]]:
        raise NotImplementedError


class AbstractTEDataset(AbstractDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        raise NotImplementedError


class AbstractQADataset(AbstractDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, Any]:
        raise NotImplementedError
