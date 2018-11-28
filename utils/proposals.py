from abc import ABC, abstractmethod
import numpy as np


class Proposal(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def propose(self):
        pass

class GaussianProposal(Proposal):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def propose(self, walker):
        return np.random.multivariate(mean=walker.get_position,
                                      cov=self.sigma,
                                      size=1)[0]
