from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from torch.utils.data import Dataset


class AbstractLogicDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()


class AbstractMCQADataset(AbstractLogicDataset):
    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Union[Tuple[str, str, List[str], int], Tuple[str, str, List[str]]]:
        raise NotImplementedError()


class AbstractTEDataset(AbstractLogicDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        raise NotImplementedError()


class AbstractQADataset(AbstractLogicDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        raise NotImplementedError()


class AbstractProofQADataset(AbstractLogicDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, str, int]:
        raise NotImplementedError()
