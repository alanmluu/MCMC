from abc import ABC, abstractmethod
from random import random


class Walker(object):

    def __init__(self):
        self.x = None

    @abstractmethod
    def walk(self, accept_ratio, x):
        pass


class SingleWalker(Walker):

    def walk(self, accept_ratio, x):
        random_draw = random()
        if accept_ratio > random_draw:
            self.x = x
            return 1
        else:
            self.x = self.x
            return 0
