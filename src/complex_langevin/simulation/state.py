# complex_langevin/simulation/state_torch.py

from complex_langevin.config import VERBOSE
from complex_langevin.utils.logging import Logger

import os
VERBOSE = os.getenv("CL_VERBOSE", "1")

class SimState(Logger):
    def __init__(self):
        self._init_logger(VERBOSE, sender = "SimState")
        self.log("initialized")