"""Global determinism utility (MASTER_SPEC Phase 0.1).

``set_global_seed`` seeds every random number generator used by the project
(python ``random``, numpy, torch CPU + CUDA) and switches cuDNN into
deterministic mode. It must be called exactly once at process start, before
any data loading or model construction.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> int:
    """Seed all RNGs and enforce deterministic backend behavior.

    Args:
        seed: Non-negative integer seed.

    Returns:
        The seed that was applied (for logging convenience).

    Raises:
        TypeError: If ``seed`` is not an ``int`` (bools are rejected too).
        ValueError: If ``seed`` is negative.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # NOTE: setting PYTHONHASHSEED inside an already-running interpreter does
    # not change str/bytes hashing for the current process; it is exported so
    # that any child processes inherit a fixed hash seed.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and (if present) all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
