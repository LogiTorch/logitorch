from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

from torch.utils.data import Dataset


class BaseLogicDataset(Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()


class AbstractMCQADataset(BaseLogicDataset):
    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Union[Tuple[str, str, List[str], int], Tuple[str, str, List[str]]]:
        raise NotImplementedError()


class AbstractTEDataset(BaseLogicDataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        raise NotImplementedError()


class AbstractQADataset(BaseLogicDataset):
    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[str, str, int],
        Tuple[str, str, str],
        Tuple[str, str, int, int],
        Tuple[List[str], str, int, List[str]],
    ]:
        raise NotImplementedError()


class AbstractProofQADataset(BaseLogicDataset):
    @abstractmethod
    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[
            Dict[str, str], Dict[str, str], List[str], List[str], List[str], List[str]
        ],
        Tuple[Dict[str, str], Dict[str, str], List[str], List[str], List[str]],
        Tuple[Dict[str, str], Dict[str, str], List[str], List[str]],
        Tuple[Dict[str, str], Dict[str, str], List[str]],
        Dict[str, Union[Optional[str], Optional[int]]],
    ]:
        raise NotImplementedError()
