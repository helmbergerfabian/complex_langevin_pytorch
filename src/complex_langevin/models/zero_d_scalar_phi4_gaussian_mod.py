from complex_langevin.utils.logging import SimLogger
from complex_langevin.config import VERBOSE
import torch

class ZeroDScalarPhi4GaussianMod(SimLogger):
    """
    PyTorch version of the old backend-kernel model:

      action      = sigma/2 * phi^2 + lamb/4 * phi^4
      mod         = - massmod/2 * phi^2
      action_mod  = action + mod

      drift_base  = sigma*phi + lamb*phi^3

      if Re(action_mod) < 0:
          out = drift_base - pull*(drift_base - massmod*phi) * exp(action_mod) / (1 + pull*exp(action_mod))
      else:
          out = drift_base - pull*(drift_base - massmod*phi) / (exp(-action_mod) + pull)

      drift = -out
    """

    def __init__(self, sigma: complex, lamb: float, massmod: complex, pull: float):
        self._init_logger(VERBOSE, sender="phi4_gaussian_mod")
        self.log("initialized")
        self.sigma = sigma
        self.lamb = lamb
        self.massmod = massmod
        self.pull = pull

    def action(self, phi: torch.Tensor) -> torch.Tensor:
        # complex action
        sigma = torch.as_tensor(self.sigma, dtype=phi.dtype, device=phi.device)
        lamb = torch.as_tensor(self.lamb, dtype=phi.real.dtype, device=phi.device)
        return 0.5 * sigma * phi**2 + 0.25 * lamb * phi**4

    def mod(self, phi: torch.Tensor) -> torch.Tensor:
        massmod = torch.as_tensor(self.massmod, dtype=phi.dtype, device=phi.device)
        return -0.5 * massmod * phi**2

    def action_mod(self, phi: torch.Tensor) -> torch.Tensor:
        return self.action(phi) + self.mod(phi)

    def drift(self, phi: torch.Tensor) -> torch.Tensor:
        sigma = torch.as_tensor(self.sigma, dtype=phi.dtype, device=phi.device)
        lamb = torch.as_tensor(self.lamb, dtype=phi.real.dtype, device=phi.device)
        massmod = torch.as_tensor(self.massmod, dtype=phi.dtype, device=phi.device)
        pull = torch.as_tensor(self.pull, dtype=phi.real.dtype, device=phi.device)

        drift_base = sigma * phi + lamb * phi**3
        a_mod = self.action_mod(phi)

        exp_a = torch.exp(a_mod)          # complex
        exp_minus_a = torch.exp(-a_mod)   # complex

        # Branch 1: Re(a_mod) < 0
        out1 = drift_base - pull * (drift_base - massmod * phi) * exp_a / (1.0 + pull * exp_a)

        # Branch 2: Re(a_mod) >= 0
        out2 = drift_base - pull * (drift_base - massmod * phi) / (exp_minus_a + pull)

        use_branch1 = (a_mod.real < 0)
        out = torch.where(use_branch1, out1, out2)

        return -out
