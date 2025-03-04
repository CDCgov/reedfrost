import functools

import numpy as np
import scipy.stats
from numpy.polynomial.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import comb


def _polynomial1(coef: float, power: int) -> Polynomial:
    """Create a polynomial with a single power

    Args:
        coef (float): coefficient
        power (int): power, must be >= 0

    Returns:
        Polynomial: coef*x^power
    """
    assert power >= 0
    return Polynomial([coef if i == power else 0.0 for i in range(power + 1)])


@functools.cache
def _kgontcharoff_polynomial(k, m) -> Polynomial:
    """
    Gontcharoff polynomial times k!, for a single value of k

    See Lefevre & Picard 1990 (doi:10.2307/1427595) equation 2.1

    Args:
        k (int): degree
        m (int): number of initial infected

    Returns:
        float: Polynomial in q
    """
    if k == 0:
        return Polynomial([1.0])
    else:
        return Polynomial([1.0]) - sum(
            _polynomial1(scipy.special.binom(k, i), (m + i) * (k - i))
            * _kgontcharoff_polynomial(i, m)
            for i in range(k)
        )


def _kgontcharoff(
    k: int | NDArray[np.int64], q: float, m: int
) -> float | NDArray[np.float64]:
    """
    Value of Gontcharoff polynomials, specific to the Lefevre & Picard
    formulation, times k!

    See Lefevre & Picard 1990 (doi:10.2307/1427595) equation 2.1

    Args:
        k (int, or int array): degree
        q (float): 1 - probability of "effective" contact
        m (int): number of initial infected

    Returns:
        float, or float array: value of the polynomial
    """
    if isinstance(k, int):
        return float(_kgontcharoff_polynomial(k=k, m=m)(q))
    else:
        return np.array([float(_kgontcharoff_polynomial(k=kk, m=m)(q)) for kk in k])


def pmf(
    k: int | NDArray[np.int64], n: int, p: float, m: int = 1
) -> float | NDArray[np.float64]:
    """
    Probability mass function for final size of a Reed-Frost outbreak.

    See Lefevre & Picard (1990) equation 3.10 for details.

    Args:
        k (int, or int array): number of total infections
        n (int): initial number susceptible
        m (int): initial number infected
        p (float): probability of "effective contact" (i.e., infection)

    Returns:
        float, or float array: pmf of the total infection distribution
    """
    q = 1.0 - p

    return comb(n, k) * q ** ((n - k) * (m + k)) * _kgontcharoff(k, q, m)


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
    result = brentq(f, 0.0, 1.0, full_output=False)
    assert isinstance(result, float)
    return result


def pmf_large(
    k: int | NDArray[np.int64], n: int, lambda_: float, i_n: int = 1
) -> float | NDArray[np.float64]:
    """Distribution of outbreak sizes of Reed-Frost outbreaks, conditioned on
    the outbreak being large.

    See Barbour & Sergey 2004 (doi:10.1016/j.spa.2004.03.013) corollary 3.4

    Args:
        k (int, or int array): number of total infections
        n (int): initial number of susceptibles
        lambda_ (float): reproduction number
        i_n (int, optional): initial number of susceptibles. Defaults to 1.

    Returns:
        float, or float array: pmf of the total infection distribution
    """
    if not lambda_ > 1.0:
        raise RuntimeWarning(
            f"Large outbreak distribution assumes lambda>1, instead saw {lambda_}"
        )
    theta = _theta_fun(i_n / n, lambda_)
    sigma = np.sqrt(theta * (1.0 - theta) / (1 - lambda_ * theta) ** 2)
    sd = np.sqrt(n) * sigma
    mean = n * (1.0 - theta)
    return scipy.stats.norm.pdf(x=k, loc=mean, scale=sd)
