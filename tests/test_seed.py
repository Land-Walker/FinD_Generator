"""Unit tests for src/utils/seed.py (Phase 0.1)."""
import random

import numpy as np
import pytest
import torch

from src.utils.seed import set_global_seed


def _draw_everything():
    return (
        random.random(),
        float(np.random.rand()),
        torch.randn(4).tolist(),
    )


def test_same_seed_reproduces_all_rngs():
    set_global_seed(123)
    first = _draw_everything()
    set_global_seed(123)
    second = _draw_everything()
    assert first == second  # bit-for-bit, no tolerance


def test_different_seeds_differ():
    set_global_seed(0)
    a = _draw_everything()
    set_global_seed(1)
    b = _draw_everything()
    assert a != b


def test_cudnn_flags_set():
    set_global_seed(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_pythonhashseed_exported():
    import os

    set_global_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_rejects_bad_seeds():
    with pytest.raises(TypeError):
        set_global_seed("0")
    with pytest.raises(TypeError):
        set_global_seed(True)
    with pytest.raises(ValueError):
        set_global_seed(-1)


def test_returns_seed():
    assert set_global_seed(5) == 5
