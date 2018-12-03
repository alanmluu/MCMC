from abc import ABC
import tensorflow as tf


class Ensemble(ABC):

    def __init__(self):
        pass


class CanonicalEnsemble(Ensemble):

    def get_ensemble_factor(self, curr_x_dens, prop_x_dens):
        return tf.divide(prop_x_dens, curr_x_dens)
