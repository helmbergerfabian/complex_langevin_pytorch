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
        # mask = self.state.alive
        if not self.state.alive.any():
            return

        # 1. drift
        self.state.drift[self.state.alive] = self.model.drift(
            self.state.phi[self.state.alive]
        )

        # 2. adaptive dt
        self.evolution.update_dt_ada()
        # self.state.dt_ada[self.state.alive] = self.evolution.compute_dt_ada(
        #     self.state.drift[self.state.alive]
        # )

        # 3. kill trajectories
        self.evolution.kill_trajs()
        
        # 4. noise
        self.evolution.update_noise()
        # self.state.noise[self.state.alive] = torch.randn(
        #     self.state.alive_count,
        #     device=self.state.device,
        #     dtype=self.state.noise.dtype,
        # )
        
        # 5. evolve
        self.evolution.update_field()
        # dt = self.state.dt_base * self.state.dt_ada[self.state.alive]

        # self.state.phi[self.state.alive] += (
        #     dt * self.state.drift[self.state.alive]
        #     + torch.sqrt(2*dt) * self.state.noise[self.state.alive]
        # )
        # self.state.langevin_time[self.state.alive] += dt

    def finish(self):
        self.log("simulation parameters: " + str({k: v for k, v in self.model.__dict__.items() if k not in {'log', "_sender"}}))
        self.log("Final alive count: {}".format(self.state.alive.sum().item()))
        self.log("Final average of langevin time: {:.4e}".format(self.state.langevin_time[self.state.alive].mean().item()))
        self.log("Final std of langevin time: {:.4e}".format(self.state.langevin_time[self.state.alive].std().item()))
        self.log("Final average of dt_ada: {:.4e}".format(self.state.dt_ada[self.state.alive].mean().item()))
        self.log("Final std of dt_ada: {:.4e}".format(self.state.dt_ada[self.state.alive].std().item()))