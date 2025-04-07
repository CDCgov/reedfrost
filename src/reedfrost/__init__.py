import functools

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from numpy.typing import NDArray


@functools.cache
def _rftp(i_next: int, s: int, i: int, p: float) -> float:
    """Reed-Frost transition probability

    Probability of there being i_next infections at the next step,
    given there are currently s susceptibles, i infected, and the
    infection probability is p.

    Args:
        i_next (int): next number of infected
        s (int): current number of susceptibles
        i (int): current number of infected
        p (float): probability of infection

    Returns:
        float: probability mass
    """
    return _pmf_binom(k=i_next, n=s, p=1.0 - (1.0 - p) ** i)


def _pmf_binom(k: int, n: int, p: float) -> float:
    """Binomial distribution pmf

    This implementation is substantially faster than scipy.stats.binom.pmf

    Args:
        k (int): number of successes
        n (int): number of trials
        p (float): success probability

    Returns:
        float: probability mass
    """
    return scipy.special.binom(n, k) * p**k * (1.0 - p) ** (n - k)


@functools.cache
def _pmf_s_inf(s_inf: int, s: int, i: int, p: float) -> float:
    """Reed-Frost final size distribution, measured in terms of numbers of
    susceptibles at infinite time

    Args:
        s_inf (int): number of susceptibles when outbreak is over
        s (int): initial number of susceptibles
        i (int): initial number of infected
        p (float): probability of infection

    Returns:
        float: probability mass
    """
    if i == 0 and s_inf == s:
        return 1.0
    elif i == 0 and s_inf != s:
        return 0.0
    elif s < s_inf:
        return 0.0
    else:
        value = sum(
            _rftp(i_next=j, s=s, i=i, p=p) * _pmf_s_inf(s_inf=s_inf, s=s - j, i=j, p=p)
            for j in range(s - s_inf + 1)
        )

        if 0.0 <= value and value <= 1.0:
            return value
        else:
            raise RuntimeError(
                f"Numerical instability for {s_inf=} {s=} {i=} {p=}: "
                f"resulting pmf value is {value}"
            )


def pmf(
    k: int | NDArray[np.integer], s: int, i: int, p: float
) -> float | NDArray[np.floating]:
    """Reed-Frost final size distribution

    Args:
        k (int | NDArray[np.integer]): number of infections beyond the initial infections
        s (int): initial number of susceptibles
        i (int): initial number of infected
        p (float): probability of infection

    Returns:
        float | NDArray[np.floating]: probability mass
    """
    if isinstance(k, (int, np.integer)):
        return _pmf_s_inf(s_inf=s - k, s=s, i=i, p=p)
    elif isinstance(k, np.ndarray):
        return np.array([_pmf_s_inf(s_inf=s - kk, s=s, i=i, p=p) for kk in k])
    else:
        raise ValueError(f"Unknown type {type(k)}")


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
