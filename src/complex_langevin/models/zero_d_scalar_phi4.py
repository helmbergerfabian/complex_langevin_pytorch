from complex_langevin.utils.logging import SimLogger
from complex_langevin.config import VERBOSE
import torch
import numpy as np

class SimModel(SimLogger):
    def __init__(self, sigma, lamb):
        self._init_logger(VERBOSE, sender = "scalar_phi4")
        self.log("initialized")
        self.sigma = sigma
        self.lamb = lamb

    def drift(self, phi: torch.Tensor) -> torch.Tensor:
        return - (self.sigma * phi + self.lamb * phi**3)

    def action(self, phi: torch.Tensor) -> torch.Tensor:
        return 1/2 * self.sigma * phi**2 + 1/4 * self.lamb * phi**4
    
    def partition_function(self) -> torch.Tensor:
        x = np.linspace(-4, 4, 1000)
        y = np.exp(-self.action(x))
        integral = np.trapezoid(y=y, x=x)
        return integral
    
    def momment(self, order):
        x = np.linspace(-4, 4, 1000)
        y = np.exp(-self.action(x)) * x**order
        integral = np.trapezoid(y=y, x=x) / self.partition_function()
        return integral