from abc import ABC


class Ensemble(ABC):

    def __init__(self):
        pass


class CanonicalEnsemble(Ensemble):

    def get_ensemble_factor(curr_x_dens, prop_x_dens):
        return prop_x_dens/curr_x_dens
