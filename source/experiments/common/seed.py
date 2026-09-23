"""Random-seed handling matching the SirenFNO-author experiment utility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU PyTorch, and all CUDA generators.

    This intentionally does not enable deterministic algorithms or alter TF32
    and backend settings because those changes are absent from the reference
    experiment.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
