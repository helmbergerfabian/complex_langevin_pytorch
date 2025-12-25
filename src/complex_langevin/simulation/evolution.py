
from complex_langevin.config import VERBOSE
from complex_langevin.utils.logging import SimLogger
import torch 

class SimEvol(SimLogger):
    """
    Evolution strategies for Complex Langevin simulations.
    """
    def __init__(self):
        self._init_logger(VERBOSE, sender = "SimEvol")
        self.log("initialized")

    def compute_dt_ada(
        self,
        drift: torch.Tensor,
        *,
        DS_MAX_LOWER: float = 1e-3,
        mean_dS_max: float = 100,
    ) -> torch.Tensor:
        """
        Adaptive timestep control.

        Parameters
        ----------
        drift : complex tensor, shape (n_seeds,)
        DS_MAX_LOWER : float
            Lower threshold for adaptive stepping.
        mean_dS_max : float
            Target maximum drift scale.

        Returns
        -------
        dt_ada : real tensor, shape (n_seeds,)
        """
        drift_abs = torch.abs(drift)

        # start with dt_ada = 1
        dt_ada = torch.ones_like(drift_abs)

        # condition: drift too large
        mask = (drift_abs > DS_MAX_LOWER) & (drift_abs > mean_dS_max)

        # apply adaptive scaling
        dt_ada[mask] = mean_dS_max / drift_abs[mask]

        return dt_ada   


    def kill_condition(self, drift):
        """
        Determine which trajectories should
        be killed based on the drift and adaptive time step.

        Args:
            drift (torch.Tensor): The drift values for the current state.
            dt_ada (torch.Tensor): The adaptive time step factors.

        Returns:
            torch.Tensor: Boolean mask indicating which trajectories to kill.
        """
        # kill trajectories with small drift mag
        norm = torch.abs(drift)
        _kill = norm < 1e-3
        return _kill