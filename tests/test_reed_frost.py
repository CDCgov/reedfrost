import numpy as np
import pytest

import reedfrost as rf


def test_pmf_2():
    """For 1 infected and 2 susceptible, do the math by hand"""

    for p in [0.1, 0.7]:
        current = rf.pmf(k=np.array([0, 1, 2]), n=2, p=p, m=1)
        expected = [(1 - p) ** 2, 2 * p * (1 - p) ** 2, p**2 + 2 * p**2 * (1 - p)]
        np.testing.assert_allclose(current, expected, atol=1e-6, rtol=0.0)


def test_pmf_vector():
    """PMF can take vectors"""
    result = rf.pmf(k=np.array([0, 1, 2]), n=2, p=0.1, m=1)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3


def test_pmf_large():
    current = rf.pmf_large(k=np.array([0, 10, 50, 90]), n=100, lambda_=1.5, i_n=1)
    np.testing.assert_allclose(
        current, [2.383925e-07, 8.889213e-06, 2.349612e-02, 1.623636e-03], rtol=1e-6
    )


def test_large_dist_warning():
    with pytest.raises(RuntimeWarning):
        rf.pmf_large(k=1, n=10, lambda_=0.5)
