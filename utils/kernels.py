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
        return np.random.multivariate_normal(mean=x,
                                             cov=self.sigma,
                                             size=1)[0]

    def get_trans_factor(self, curr_x, prop_x):
        return 1

"""
class Hamiltonian(Kernel):

    def __init__(self, steps, eps):
        self.steps = steps
        self.eps = eps

    def propose(self, x, distribution):

"""
