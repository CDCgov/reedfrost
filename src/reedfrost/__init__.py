import functools
import math

import numpy as np
import scipy.optimize
import scipy.stats


@functools.cache
def _gontcharoff(k: int, q: float, m: int) -> float:
    """
    Gontcharoff polynomials, specific to the Lefevre & Picard
    formulations of Reed-Frost final outbreak size pmf calculations

    See Lefevre & Picard 1990 (doi:10.2307/1427595) equation 2.1

    Args:
        k (int): degree
        q (float): 1 - probability of "effective" contact
        m (int): number of initial infected

    Returns:
        float: value of the polynomial
    """
    if k == 0:
        return 1.0
    else:
        return 1.0 / math.factorial(k) - sum(
            [
                q ** ((m + i) * (k - i)) / math.factorial(k - i) * _gontcharoff(i, q, m)
                for i in range(0, k)
            ]
        )


def pmf(k: int, n: int, p: float, m: int = 1) -> float:
    """
    Probability mass function for final size of a Reed-Frost outbreak

    See Lefevre & Picard 1990 (doi:10.2307/1427595) equation 3.10

    Args:
        k (int): number of total infections
        n (int): initial number susceptible
        m (int): initial number infected
        p (float): probability of "effective contact" (i.e., infection)

    Returns:
        float: pmf of the total infection distribution
    """
    q = 1.0 - p
    return (
        math.factorial(n)
        / math.factorial(n - k)
        * q ** ((n - k) * (m + k))
        * _gontcharoff(k, q, m)
    )


def _theta_fun(w: float, lambda_: float) -> float:
    """Function for theta_n as per Barbour & Sergey 2004

    Args:
        w (float): variable for root-finding
        lambda_ (float): reproduction number

    Returns:
        float: root
    """

    def f(t):
        if t == 0.0:
            return np.inf
        else:
            return t - np.log(t) / lambda_ - (1.0 + w)

    # do type checking here because type hinting gets confused about whether
    # this results a tuple or a float
    result = scipy.optimize.brentq(f, 0.0, 1.0, full_output=False)
    assert isinstance(result, float)
    return result


def dist_large(n: int, lambda_: float, i_n: int = 1):
    """Distribution of outbreak sizes, given a large outbreak

    See Barbour & Sergey 2004 (doi:10.1016/j.spa.2004.03.013) corollary 3.4

    Args:
        n (int): initial number of susceptibles
        lambda_ (float): reproduction number
        i_n (int, optional): initial number of susceptibles. Defaults to 1.

    Returns:
        scipy.stats.norm: RV object

    Examples:
        dist_large(100, 1.5, 1).pdf(np.linspace(0, 100))
    """
    if not lambda_ > 1.0:
        raise RuntimeWarning(
            f"Large outbreak distribution assumes lambda>1, instead saw {lambda_}"
        )
    theta = _theta_fun(i_n / n, lambda_)
    sigma = np.sqrt(theta * (1.0 - theta) / (1 - lambda_ * theta) ** 2)
    sd = np.sqrt(n) * sigma
    mean = n * (1.0 - theta)
    return scipy.stats.norm(loc=mean, scale=sd)
