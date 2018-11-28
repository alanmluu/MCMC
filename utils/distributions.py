import tensorflow as tf
import numpy as np
import support_functions


class Gaussian(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.i_sigma = np.linalg.inv(np.copy(sigma))

    def get_energy_fn(self):

        def energy_fn(x):
            return support_functions.gaussian_energy(x, self.mu, self.i_sigma)

        return energy_fn
