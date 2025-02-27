import pytest
from pytest import approx

import reedfrost as rf


def test_pmf_2():
    """For 1 infected and 2 susceptible, do the math by hand"""

    for p in [0.1, 0.7]:
        assert rf.pmf(0, 2, p, m=1) == approx((1 - p) ** 2, abs=1e-6)
        assert rf.pmf(1, 2, p, m=1) == approx(2 * p * (1 - p) ** 2, abs=1e-6)
        assert rf.pmf(2, 2, p, m=1) == approx(p**2 + 2 * p**2 * (1 - p), abs=1e-6)


def test_large_dist_warning():
    with pytest.raises(RuntimeWarning):
        rf.dist_large(n=10, lmbda=0.5)
