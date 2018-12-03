from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, walker):
        pass


class ConstantInitializer(Initializer):

    def __init__(self, x):
        self.x = x

    def initialize(self, walker):
        walker.x = self.x
