from abc import ABC, abstractmethod


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
        walker.set_position(self.x)
