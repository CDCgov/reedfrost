import numpy as np
import scipy.stats

from reedfrost import ChainBinomial


def test_pmf_binom():
    """Homegrown binomial pmf should be equal to gold standard"""
    n = 10
    k = np.array(range(n + 1))
    p = 0.02
    np.testing.assert_almost_equal(
        scipy.stats.binom.pmf(k=k, n=n, p=p),
        [ChainBinomial._pmf_binom(k=kk, n=n, p=p) for kk in k],
    )
