from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def propose(self):
        pass

class GaussianKernel(Kernel):

    def __init__(self, sigma):
        self.sigma = sigma

    def propose(self, x, distribution):
        return np.random.multivariate(mean=cx,
                                      cov=self.sigma,
                                      size=1)[0]

class Hamiltonian(Kernel):

    def __init__(self, steps, eps):
        self.steps = steps
        self.eps = eps

    def propose(self, x, distribution):
        
