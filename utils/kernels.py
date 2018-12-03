from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


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
        self.sigma_n = sigma.numpy()

    def propose(self, x, distribution):
        x_n = x.numpy()[0]
        proposal_n = np.random.multivariate_normal(mean=x_n,
                                                    cov=self.sigma_n,
                                                    size=1)[0]
        return tf.constant([proposal_n], dtype="float32")

    def get_trans_factor(self, curr_x, prop_x):
        return tf.constant([1], dtype="float32")


class Hamiltonian(Kernel):

    def __init__(self, steps, eps):
        self.steps = steps
        self.eps = eps
