
from complex_langevin.config import VERBOSE, DS_MAX_LOWER, mean_dS_max
from complex_langevin.utils.logging import SimLogger
from complex_langevin.simulation.state import SimState
from complex_langevin.models.zero_d_scalar_phi4 import SimModel
import torch 

class SimEvol(SimLogger):
    """
    Evolution strategies for Complex Langevin simulations.
    """
    def __init__(self, state: SimState, model: SimModel):
        self._init_logger(VERBOSE, sender = "SimEvol")
        self.log("initialized")
        
        self.state = state
        self.model = model

    def update_dt_ada(self):
        """
        Update the per-trajectory adaptive timestep factors.

        For all currently alive trajectories, the adaptive factor dt_ada is
        reset to 1. Trajectories with sufficiently large drift magnitude
        (above DS_MAX_LOWER and the current mean_dS_max) receive a reduced
        timestep according to

            dt_ada = mean_dS_max / |drift|.

        Dead trajectories are not modified.

        This method performs only the mechanical update of dt_ada; the
        interpretation of small dt_ada values (e.g. trajectory killing or
        rejection) is handled elsewhere in the evolution logic.
        """
        drift_abs = torch.abs(self.state.drift)
        self.state.dt_ada[self.state.alive] = 1

        mask = self.state.alive & (drift_abs > DS_MAX_LOWER) & (drift_abs > mean_dS_max)        
        self.state.dt_ada[mask] = mean_dS_max / drift_abs[mask]

    def compute_dt_ada(
        self,
        drift: torch.Tensor,
        *,
        DS_MAX_LOWER: float = 1e-3,
        mean_dS_max: float = 5,
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

    def update_field(self):
        dt = self.state.dt_base * self.state.dt_ada[self.state.alive]

        self.state.phi[self.state.alive] += (
            dt * self.state.drift[self.state.alive]
            + torch.sqrt(2*dt) * self.state.noise[self.state.alive]
        )
        self.state.langevin_time[self.state.alive] += dt


    def update_noise(self):
        self.state.noise[self.state.alive] = \
        torch.randn(self.state.alive_count,device=self.state.device,
                    dtype=self.state.noise.dtype,
        )       

    
    def update_drift(self):
        self.state.drift[self.state.alive] = \
        self.model.drift(self.state.phi[self.state.alive])


    def kill_trajs(self):
        """
        Kill trajectories based on adaptive timestep criterion.
        Evolution logic only; state just applies the new mask.
        """
        alive = self.state.alive

        # kill condition, restricted to alive seeds
        kill = (self.state.dt_ada < 1e-3) & alive

        if not kill.any():
            return

        new_alive = alive.clone()
        new_alive[kill] = False

        self.state.alive = new_alive