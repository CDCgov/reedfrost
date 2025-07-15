import numpy as np
import scipy.stats

import reedfrost as rf
from reedfrost import ReedFrost


def test_pmf_binom():
    """Homegrown binomial pmf should be equal to gold standard"""
    n = 10
    k = np.array(range(n + 1))
    p = 0.02
    np.testing.assert_almost_equal(
        scipy.stats.binom.pmf(k=k, n=n, p=p),
        [ReedFrost._pmf_binom(k=kk, n=n, p=p) for kk in k],
    )


def test_pmf_2():
    """For 1 infected and 2 susceptible, do the math by hand"""
    for p in [0.1, 0.7]:
        x = ReedFrost(s0=2, p=p, i0=1)
        current = [x.prob_final_i_cum_extra(k) for k in [0, 1, 2]]
        expected = [(1 - p) ** 2, 2 * p * (1 - p) ** 2, p**2 + 2 * p**2 * (1 - p)]
        np.testing.assert_allclose(current, expected, atol=1e-6, rtol=0.0)


def test_pmf_snapshot():
    """Check for a few known values"""
    current = np.array(
        [
            ReedFrost(s0=10, i0=1, p=0.1).prob_final_i_cum_extra(0),
            ReedFrost(s0=11, i0=2, p=0.2).prob_final_i_cum_extra(1),
            ReedFrost(s0=12, i0=3, p=0.3).prob_final_i_cum_extra(3),
        ]
    )
    expected = np.array([3.486784e-01, 4.902243e-03, 5.321873e-07])
    np.testing.assert_allclose(current, expected, rtol=1e-6)


def test_simulate():
    """Simulate a Reed-Frost outbreak"""
    s = 10
    i = 1
    p = 0.1
    rng = np.random.default_rng(42)
    result = rf.simulate(s=s, i=i, p=p, rng=rng)
    assert isinstance(result, np.ndarray)
    assert len(result) == s + 1
    assert result[0] == i  # initial infected
    assert np.all(result[1:] >= 0)  # no negative infections
