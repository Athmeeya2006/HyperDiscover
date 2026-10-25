import numpy as np
import pytest
import torch

torch.set_num_threads(2)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
