from complex_langevin.utils.logging import SimLogger
from complex_langevin.config import VERBOSE
import torch

class SimModel(SimLogger):
    def __init__(self, sigma, lamb):
        self._init_logger(VERBOSE, sender = "scalar_phi4")
        self.log("initialized")
        self.sigma = sigma
        self.lamb = lamb

    def drift(self, phi: torch.Tensor) -> torch.Tensor:
        print(- (self.sigma * phi + self.lamb * phi**3))
        return - (self.sigma * phi + self.lamb * phi**3)