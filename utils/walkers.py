from abc import ABC, abstractmethod


class Walker(ABC):

    def __init__(self, dim, initializer, proposer, acceptor):
        self.dim = dim
        self.initializer = initializer
        self.proposal = proposer
        self.acceptor = acceptor

    @abstractmethod
    def set_position(self, x):
        pass

    @abstractmethod
    def get_position(self):
        return self.x


class SingleWalker(Walker):

    def set_position(self, x):
        self.x = x

    def get_position(self):
        return self.x
