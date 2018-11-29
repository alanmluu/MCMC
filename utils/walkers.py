from abc import ABC, abstractmethod
from random import random


class Walker(ABC):

    def __init__(self):
        self.x = None

    def set_position(self, x):
        self.x = x

    def get_position(self):
        return self.x

    @abstractmethod
    def walk(self, accept_ratio, x):
        pass


class SingleWalker(Walker):

    def walk(self, accept_ratio, x):
        random_draw = random()
        if accept_ratio > random_draw:
            self.set_position(x)
        else:
            self.set_position(self.x)
