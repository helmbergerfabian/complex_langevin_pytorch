# complex_langevin/simulation/state_torch.py

from complex_langevin.config import VERBOSE
from complex_langevin.utils.logging import Logger

from complex_langevin.config import CL_REAL, CL_COMPLEX
import torch

class SimState(Logger):
    def __init__(
        self,
        n_seeds: int | None = None,
        dt_base: float | None = None,
        device: str | torch.device | None = None,
    ):
        # initialize logger
        self._init_logger(VERBOSE, sender = "SimState")
        self.log("initialized")

        # choose device, which is CPU or GPU
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        

        # number of seeds
        self.n_seeds = n_seeds or int(1e4)

        # base time step per seed, later mulitplied by adaptive factor
        self.dt_base = torch.tensor(
            dt_base if dt_base is not None else 1e-4,
            dtype=CL_REAL,
            device=self.device,
        )

        # field values per seed
        self.phi = torch.zeros(
            self.n_seeds,
            dtype=CL_COMPLEX,
            device=self.device,
        )

        # drift array
        self.drift = torch.zeros_like(self.phi)

        # real valued noise per seed
        self.noise = torch.zeros(
            self.n_seeds,
            dtype=CL_REAL,
            device=self.device,
        )

        # adaptive time step factors per seed
        self.dt_ada = torch.ones(
            self.n_seeds,
            dtype=CL_REAL,
            device=self.device,
        )

        self.langevin_time = torch.zeros(
            self.n_seeds,
            dtype=CL_REAL,
            device=self.device,
        )