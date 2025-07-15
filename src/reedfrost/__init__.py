import functools

import numpy as np
import scipy.special
from numpy.typing import NDArray


class ReedFrost:
    def __init__(self, s0: int, p: float, i0: int = 1):
        assert s0 >= 0
        assert 0.0 <= p <= 1.0
        assert i0 >= 0
        self.s0 = s0
        self.p = p
        self.i0 = i0

    def prob_final_s(self, s_inf: int) -> float:
        """Probability of a certain final number of susceptibles"""
        assert s_inf >= 0
        return self.prob_state(s=s_inf, i=0, t=self.s0 + 1)

    def prob_final_i_cum_extra(self, k: int) -> float:
        """
        Probability of a certain number of infection beyond the initial infection(s)
        """
        assert k >= 0
        return self.prob_final_s(s_inf=self.s0 - k)

    def prob_final_i_cum(self, i_cum: int) -> float:
        """
        Probability of a certain number of total infections, including the initial infection(s)
        """
        assert i_cum >= 0
        if i_cum < self.i0:
            return 0.0
        else:
            return self.prob_final_i_cum_extra(k=i_cum - self.i0)

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
        assert t >= 0
        assert i >= 0
        assert s >= 0

        if t == 0 and s == self.s0 and i == self.i0:
            return 1.0
        elif t == 0:
            return 0.0
        elif s + i > self.s0:
            return 0.0
        elif t > 0:
            return sum(
                [
                    self._tp(i, s + i, ip, p=self.p) * self.prob_state(s + i, ip, t - 1)
                    for ip in range((self.s0 + self.i0) - (s + i) + 1)
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

    def simulate(
        self,
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
        it = np.zeros(self.s0 + 1, dtype=np.int64)

        if self.i0 == 0:
            return it

        it[0] = self.i0
        s = self.s0

        for t in range(self.s0):
            if it[t] == 0:
                break

            next_i = rng.binomial(n=s, p=1.0 - (1.0 - self.p) ** it[t])
            it[t + 1] = next_i
            s = s - next_i

        return it
