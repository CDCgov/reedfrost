import functools

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from numpy.typing import NDArray


class ReedFrost:
    def __init__(self, s0: int, p: float, i0: int = 1):
        assert s0 >= 0
        assert 0.0 <= p <= 1.0
        assert i0 >= 0
        self.s0 = s0
        self.p = p
        self.i0 = i0

    def prob_final_size(self, s_inf: int) -> float:
        """Probability of being in state (s_inf, 0) at infinite time

        Args:
            s_inf (int): number of susceptibles when outbreak is over

        Returns:
            float: probability mass
        """
        return self.prob_state(s=s_inf, i=0, t=self.s0 + 1)

    @functools.cache
    def prob_state(self, s: int, i: int, t: int) -> float:
        """Probability of being in state (s, i) at time t

        Args:
            s (int): number of susceptibles
            i (int): number of infected
            t (int): generation

        Returns:
            float: probability mass
        """
        if t == 0:
            if s == self.s0 and i == self.i0:
                return 1.0
            else:
                return 0.0
        elif t > 0:
            return sum(
                [
                    self._tp(i, s + i, ip, p=self.p) * self.prob_state(s + i, ip, t - 1)
                    for ip in range(self.s0 + 1)
                    if self.prob_state(s + i, ip, t - 1) > 0.0
                ]
            )
        else:
            raise ValueError(f"Negative generation {t}")

    @classmethod
    @functools.cache
    def _tp(cls, i, si, ip, p: float) -> float:
        return cls._pmf_binom(k=i, n=si, p=1.0 - (1.0 - p) ** ip)

    @staticmethod
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
    return ReedFrost._pmf_binom(k=i_next, n=s, p=1.0 - (1.0 - p) ** i)


def simulate(
    s: int,
    i: int,
    p: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> NDArray[np.integer]:
    """Simulate a Reed-Frost outbreak

    Args:
        s (int): initial number of susceptibles
        i (int): initial number of infected
        p (float): probability of infection

    Returns:
        NDArray[np.integer]: number of infected in each generation
    """
    # time series of infections, starting with the initial infected,
    # is at most of length s + 1
    it = np.zeros(s + 1, dtype=np.int64)

    if i == 0:
        return it

    it[0] = i

    for t in range(s):
        if it[t] == 0:
            break

        next_i = rng.binomial(n=s, p=1.0 - (1.0 - p) ** it[t])
        it[t + 1] = next_i
        s = s - next_i

    return it
