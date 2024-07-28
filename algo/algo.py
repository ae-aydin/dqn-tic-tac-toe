from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
