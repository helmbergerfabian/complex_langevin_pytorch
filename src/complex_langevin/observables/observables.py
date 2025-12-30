from complex_langevin.utils.logging import SimLogger
from complex_langevin.simulation.runner import SimRunner
from complex_langevin.simulation.state import SimState
from complex_langevin.models.zero_d_scalar_phi4 import SimModel
from complex_langevin.config import CL_REAL, VERBOSE, CL_COMPLEX

import torch
class SimObs(SimLogger):
    def __init__(
        self,
        state: SimState,
        model: SimModel,
        runner: SimRunner,
        name: str,
        auto_corr: float,
        order: int,
        target_blocks: int = 20
    ):
        self._init_logger(VERBOSE, sender = "SimObs")
        self.log(f"{name} initialized")

        self.state = state
        self.model = model
        self.runner = runner

        self.name = name
        self.auto_corr = auto_corr
        self.order = order

        self.rolling_accumulator = RollingComplexAccumulator(
            self.state.n_seeds, self.state.device, self, target_blocks = target_blocks
        )

        self.block_stack = []  # block-level observables (complex)

        self.last_rolling_meas = torch.zeros(
            self.state.n_seeds,
            dtype=CL_REAL,
            device=self.state.device,
        )

        self.last_block_flush = torch.zeros(
            self.state.n_seeds,
            dtype=CL_REAL,
            device=self.state.device,
        )

        self._cold_rolling = torch.ones(
            self.state.n_seeds,
            dtype=torch.bool,
            device=self.state.device,
        )

        self._cold_block = torch.ones(
            self.state.n_seeds,
            dtype=torch.bool,
            device=self.state.device,
        )

        self.target_blocks = target_blocks

        self.blocks_done = torch.zeros(
            self.state.n_seeds,
            dtype=torch.int64,
            device=self.state.device,
        )

    @property
    def cold_rolling(self):
        mask = self.state.alive
        self._cold_rolling[:] = False
        self._cold_rolling[mask] = (
            self.state.langevin_time[mask]
            - self.last_rolling_meas[mask]
            >= self.state.dt_base
        )
        return self._cold_rolling

    @property
    def cold_block(self):
        mask = self.state.alive
        self._cold_block[:] = False
        self._cold_block[mask] = (
            self.state.langevin_time[mask]
            - self.last_block_flush[mask]
            >= self.auto_corr
        )
        return self._cold_block
    
    @property
    def daq_active(self):
        """
        Seeds that are still collecting data for this observable.
        """
        return self.state.alive & (self.blocks_done < self.target_blocks)
    
    @property
    def is_done(self):
        return not self.daq_active.any()
    
    def measure_observable(self):
        return self.state.phi ** self.order

    def observe(self):

        # rolling measurements
        # find those seeds, that have passed dt_base since last observation
        mask = self.cold_rolling & self.daq_active

        # if there are any, observe them
        if mask.any():
            obs = self.measure_observable()

            # pass the new data to the rolling accumulator
            self.rolling_accumulator.update_rolling(obs, mask)

            # for those seedsobserved, write the time of observation
            self.last_rolling_meas[mask] = self.state.langevin_time[mask]


        # reduce rolling observations
        # find those seeds, that have passed auto_corr since last reduction
        mask = (
            self.cold_block
            & self.daq_active
            & (self.rolling_accumulator.count_rolling > 0)
        )     

        # if there are any, reduce them
        if mask.any():
            self.rolling_accumulator.update_reduced(mask)
            self.rolling_accumulator.reset(mask)

            self.blocks_done[mask] += 1          # NEW
            self.last_block_flush[mask] = self.state.langevin_time[mask]

    
    def finish(self):
        self.rolling_accumulator.block_means = (
            self.rolling_accumulator.block_means.cpu()
        )
        self.rolling_accumulator.reduction_times = (
            self.rolling_accumulator.reduction_times.cpu()
        )




class RollingComplexAccumulator(SimLogger):
    def __init__(self, n_seeds, device, observable: SimObs, target_blocks: int = 20):
        self._init_logger(VERBOSE, "RollingAcc")
        self.observable = observable
        self.target_blocks = target_blocks

        self.sum_x  = torch.zeros(n_seeds, device=device)
        self.sum_y  = torch.zeros(n_seeds, device=device)
        self.sum_xx = torch.zeros(n_seeds, device=device)
        self.sum_yy = torch.zeros(n_seeds, device=device)
        self.sum_xy = torch.zeros(n_seeds, device=device)

        self.count_rolling  = torch.zeros(n_seeds, dtype=torch.int64, device=device)
        self.count_reduced  = torch.zeros(n_seeds, dtype=torch.int64, device=device)

        self.block_means = torch.full(
            (self.target_blocks, n_seeds),
            float("nan"),
            device=device,
            dtype=CL_COMPLEX
        )
        self.reduction_times = torch.zeros((self.target_blocks, n_seeds), device=device)

    def update_rolling(self, z: torch.Tensor, mask: torch.Tensor):
        """
        Add new measurements z[mask] to the rolling block.
        """
        x = z.real
        y = z.imag

        self.sum_x[mask]  += x[mask]
        self.sum_y[mask]  += y[mask]
        self.sum_xx[mask] += x[mask] * x[mask]
        self.sum_yy[mask] += y[mask] * y[mask]
        self.sum_xy[mask] += x[mask] * y[mask]
        self.count_rolling[mask]  += 1

    def update_reduced(self, mask: torch.Tensor):

        valid = mask & (self.count_rolling > 0)

        if valid.any():
            rows = torch.nonzero(valid, as_tuple=True)[0]
            nums = self.count_rolling[rows]

            mean_x = self.sum_x[rows] / nums
            mean_y = self.sum_y[rows] / nums
            
            cols = self.count_reduced[rows]
            self.block_means[cols, rows] = mean_x +1j * mean_y
            self.reduction_times[cols, rows] = self.observable.state.langevin_time[valid]
            self.count_reduced[valid] += 1
    
    def reset(self, mask):
        self.sum_x[mask]  = 0
        self.sum_y[mask]  = 0
        self.sum_xx[mask] = 0
        self.sum_yy[mask] = 0
        self.sum_xy[mask] = 0
        self.count_rolling[mask]  = 0

    def get_block_matrix(self):
        """
        Returns:
            block_means: (n_blocks, n_seeds) with NaNs
            reduction_times: same shape
        """
        return self.block_means[:self.observable.target_blocks], \
            self.reduction_times[:self.observable.target_blocks]

import torch
from complex_langevin.config import CL_COMPLEX


class DSEMoment(SimObs):
    def __init__(
        self,
        *,
        pullback: float,
        mass_mod: float,
        **kwargs,
    ):
        """
        DSE-specific observable.

        All DAQ, blocking, async logic is inherited from SimObs.
        Only the observable definition is changed.
        """
        super().__init__(**kwargs)

        self.pullback = pullback
        self.mass_mod = mass_mod

        self.log(
            f"DSEMoment initialized "
            f"(order={self.order}, pullback={pullback}, mass_mod={mass_mod})"
        )

    # ------------------------------------------------------------------
    # ONLY THING THAT CHANGES
    # ------------------------------------------------------------------
    def measure_observable(self):
        phi = self.state.phi

        # Effective action with modified mass

        return self.order  * phi**(self.order-1) - phi**self.order * (self.model.sigma*phi + self.model.lamb*phi**3)