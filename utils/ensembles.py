from abc import ABC, abstractmethod
from random import random


class Ensemble(ABC):

    def __init__(self):
        pass


class CanonicalEnsemble(Ensemble):

    def get_ens_factor(curr_x_dens, prop_x_dens):
        return prop_x_dens/curr_x_dens
