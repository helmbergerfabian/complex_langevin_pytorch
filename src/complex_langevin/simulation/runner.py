from complex_langevin.simulation.state import SimState
from complex_langevin.simulation.evolution import SimEvol
from complex_langevin.models.zero_d_scalar_phi4 import SimModel
from complex_langevin.utils.logging import SimLogger
from complex_langevin.config import VERBOSE


import torch 

class SimRunner(SimLogger):
    """
    PyTorch-based Complex Langevin simulation loop.
    """

    def __init__(self, state: SimState, model: SimModel, evolution: SimEvol):
        self._init_logger(VERBOSE, sender = "SimRunner")
        self.log("initialized")
        self.state = state
        self.model = model
        self.evolution = evolution


    def step(self):
        mask = self.state.alive
        if not mask.any():
            return

        # 1. drift
        self.state.drift[mask] = self.model.drift(
            self.state.phi[mask]
        )

        # 2. adaptive dt
        self.state.dt_ada[mask] = self.evolution.compute_dt_ada(
            self.state.drift[mask]
        )

        # 3. kill trajectories
        kill = self.evolution.kill_condition(
            self.state.drift[mask],
            self.state.dt_ada[mask],
        )
        
        mask = self.state.alive
        new_alive = self.state.alive.clone()
        new_alive[mask] &= ~kill
        self.state.alive = new_alive

        # # refresh mask after killing
        # mask = self.state.alive
        # if not mask.any():
        #     return

        # 4. noise
        self.state.noise[mask] = torch.randn(
            mask.sum(),
            device=self.state.device,
            dtype=self.state.noise.dtype,
        )

        # 5. evolve
        dt = self.state.dt_base * self.state.dt_ada[mask]

        self.state.phi[mask] += (
            dt * self.state.drift[mask]
            + torch.sqrt(dt) * self.state.noise[mask]
        )

        self.state.langevin_time[mask] += dt
        # self.state.global_step += 1
