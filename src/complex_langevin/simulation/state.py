# complex_langevin/simulation/state_torch.py

from complex_langevin.config import VERBOSE
from complex_langevin.utils.logging import SimLogger

from complex_langevin.config import CL_REAL, CL_COMPLEX
import torch

class SimState(SimLogger):
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

        # langevin time per seed
        self.langevin_time = torch.zeros(
            self.n_seeds,
            dtype=CL_REAL,
            device=self.device,
        )
        
        # alive mask per seed
        self._alive = torch.ones(
            self.n_seeds,
            dtype=torch.bool,
            device=self.device,
        )

    @property
    def alive(self) -> torch.Tensor:
        """Boolean mask of alive trajectories (read-only view)."""
        return self._alive

    
    @alive.setter
    def alive(self, new_alive: torch.Tensor) -> None:
        """
        Replace alive mask.
        Evolution logic decides what dies; state only applies it.
        """
        if not isinstance(new_alive, torch.Tensor):
            raise TypeError("alive must be a torch.Tensor")

        if new_alive.dtype != torch.bool:
            raise TypeError("alive mask must be boolean")

        if new_alive.shape != self._alive.shape:
            raise ValueError("alive mask has wrong shape")

        if new_alive.device != self._alive.device:
            raise ValueError("alive mask on wrong device")

        self._alive = new_alive


    @property
    def alive_count(self):
        return self.alive.sum()