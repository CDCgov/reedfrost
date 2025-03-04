# Implementation

There are at least two methods for computing the final size distribution of Reed-Frost outbreaks, including a dynamic programming approach and the approach of Lefevre & Picard.

The dynamic programming approach is more numerically stable but has $\mathcal{O}(n^3)$ complexity, while the Lefevre & Picard approach has $\mathcal{O}(n)$ complexity. In practice, we found that the numerical instability is more limiting than the computational time, so we currently opt for the dynamic programming approach.

## Dynamic programming approach

The probability mass function $f$, in terms of the number $s_\infty$ of susceptibles who who remain at the end of the outbreak, is:

```math
f_{S_\infty}(s_\infty; s, i, p) = \begin{cases}
  1 & i = 0 \text{ and } s_\infty = s \\
  0 & i = 0 \text{ and } s_\infty \neq s \\
  0 & s < s_\infty \\
  \sum_{j=0}^{s-s_\infty} f_\mathrm{binom}\left(j; s, 1-(1-p)^i\right) \times f_{S_\infty}(s_\infty; s-j, j, p) & \text{otherwise}
\end{cases}
```

where $s$ and $i$ are the starting numbers of susceptibles and infected and $p$ is the probability of infection. In other words:

- If the outbreak is over, the starting and ending number of remaining susceptibles must be the same.
- There cannot be more ending susceptibles than starting susceptibles.
- In other cases, note that we can advance one generation, allowing a number of infections between 0 and the maximum possible ($s - s_\infty$). The probability of ending at $s_\infty$ is the probability of each of those numbers of infections times the probability of going from that updated state to the final state.

The probability mass function, in terms of the number $k$ of infections beyond the initial $i$, is:

```math
f(k; s, i, p) = f_{S_\infty}(s - k; s, i, p)
```

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

In general, the Gontcharoff polynomials are (cf. equation 2.1):

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
