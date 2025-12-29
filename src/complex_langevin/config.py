### src/complex_langevin/config.py
import torch
import os

CL_PRECISION = os.getenv("CL_PRECISION", "double").lower()

if CL_PRECISION == "double":
    CL_REAL = torch.float64
    CL_COMPLEX = torch.complex128
    CL_INT = torch.int64
elif CL_PRECISION == "single":
    CL_REAL = torch.float32
    CL_COMPLEX = torch.complex64
    CL_INT = torch.int32
else:
    raise ValueError(f"Unknown CL_PRECISION={CL_PRECISION}")


VERBOSE = os.getenv("CL_VERBOSE", "1") == "1"

DS_MAX_LOWER = 1e-3
mean_dS_max = 100