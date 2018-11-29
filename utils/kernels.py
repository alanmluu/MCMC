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

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def propose(self, walker):
        return np.random.multivariate(mean=walker.get_position,
                                      cov=self.sigma,
                                      size=1)[0]
