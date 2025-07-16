# Implementation

There are at least two methods for computing the final size distribution of Reed-Frost outbreaks, including a dynamic programming approach and the approach of Lefevre & Picard.

The dynamic programming approach is more numerically stable but has $\mathcal{O}(n^3)$ complexity, while the Lefevre & Picard approach has $\mathcal{O}(n)$ complexity. In practice, we found that the numerical instability is more limiting than the computational time, so we currently opt for the dynamic programming approach.

## Dynamic programming approach

See first the [model description](chain_binomial.md).

Let $P(s, i, t)$ be the probability of being in state $(s, i)$ in generation $t$. Begin from an initial state, setting $P(s_0, i_0, 0) = 1$. Then iteratively generate:

```math
P(s, i, t) = \sum_{i'=0}^{s_0-s-i} f_\mathrm{Binom}(i; s+i, \pi(i') ) \cdot P(s + i, i', t - 1)
```

### Final size

By time $t=s_0+1$, we are guaranteed to have $i=0$, because the longest time to extinction will occur when there is 1 infection in each generation. Thus, $P(s_\infty, 0, s_0+1)$ is the distribution of final sizes, parameterized by the number of remaining susceptibles $s_\infty$. The cumulative number of infections is $i_0 + (s_0 - s_\infty)$.

## Lefevre & Picard approach

See [Lefevre & Picard (1990)](https://www.doi.org/10.2307/1427595) equation 3.10:

```math
f(k; n, m, p) = n_{(k)} q^{(n-k)(m+k)} G_k(1 | q^{m+i}, i \in \mathbb{N})
```

where:

- $n_{(k)} = n!/(n-k)!$ is the [falling factorial](https://en.wikipedia.org/wiki/Falling_and_rising_factorials),
- $k$ is the number of infections beyond the initial infections,
- $n$ is the initial number susceptible (called $S_0$ above),
- $m$ is the initial number infected (called $I_0$ above),
- $p$ is the probability of effective contact (i.e., that any infected infects any susceptible in a time step),
- $G_k$ are the Gontcharoff polynomials.

### Gontcharoff polynomials

In general, the Gontcharoff polynomials are (cf. Lefevre & Picard equation 2.1):

```math
G_k(x | U) = \begin{cases}
  1 & k = 0 \\
  \frac{x^k}{k!} - \sum_{i=0}^{k-1} \frac{u_i^{k-i}}{(k-i)!} G_i(x | U) & k > 0
\end{cases}
```

For convenience, define $g(k, q, m) \equiv G_k(1 | q^{m+i}, i \in \mathbb{N})$, a version of the Gontcharoff polynomials that are specific to this application, so that:

```math
g(k, q, m) = \begin{cases}
  1 & k = 0 \\
  \frac{1}{k!} - \sum_{i=0}^{k-1} \frac{1}{(k-i)!} q^{(m+i)(k-i)} g(i, q, m) & k > 0
\end{cases}
```

### Preventing overflow

In practice, the formulation for $f$ and $g$ above lead to numerical overflow. To avoid this, define $h(k, q, m) = k! \times g(k, q, m)$ such that:

```math
h(k, q, m) = \begin{cases}
  1 & k = 0 \\
  1 - \sum_{i=0}^{k-1} \binom{k}{i} q^{(m+i)(k-i)} h(i, q, m) & k > 0
\end{cases}
```

and

```math
f(k; n, m, p) = \binom{n}{k} q^{(n-k)(m+k)} h(k, q, m)
```

The use of binomial coefficients, rather than pure factorials, increases the range of values $k$ and $n$ before causing overflow.

### Numerical instability

In practice, numerical instability is a more pressing problem that leads to values of $f$ greater than unity. The individual $h(k, q, m)$ are very small, so that the sum over the $h(i, q, m)$ terms is a sum of many small numbers that come to nearly unity.

The numerical instability is most pronounced for $q \approx 1$, i.e., $p \ll 1$.
