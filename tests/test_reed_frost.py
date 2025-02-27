import numpy as np
import pytest
from pytest import approx

import reedfrost as rf


def test_pmf_2():
    """For 1 infected and 2 susceptible, do the math by hand"""

    for p in [0.1, 0.7]:
        assert rf.pmf(0, 2, p, m=1) == approx((1 - p) ** 2, abs=1e-6)
        assert rf.pmf(1, 2, p, m=1) == approx(2 * p * (1 - p) ** 2, abs=1e-6)
        assert rf.pmf(2, 2, p, m=1) == approx(p**2 + 2 * p**2 * (1 - p), abs=1e-6)


def test_pmf_vector():
    """PMF can take vectors"""
    result = rf.pmf(k=np.array([0, 1, 2]), n=2, p=0.1, m=1)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3


def test_large_dist_warning():
    with pytest.raises(RuntimeWarning):
        rf.pmf_large(k=1, n=10, lambda_=0.5)
