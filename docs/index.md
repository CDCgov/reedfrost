# The Reed-Frost model

In [the model](https://en.wikipedia.org/wiki/Reed%E2%80%93Frost_model), there are susceptible and infected people. At each time step, each infected person has an independent probability $p$ to infect each susceptible person. Infected people are removed at each time step.

Let $S_0$ and $I_0$ be the initial numbers susceptible and infected. Then the number infected and susceptible at each subsequent time point is:

```math
\begin{align*}
I_{t+1} &\sim \mathrm{Binomial}\left[S_t; 1-(1-p)^{I_t}\right] \\
S_{t+1} &= S_t - I_{t+1}
\end{align*}
```

## Final size distributions

The probability mass function for the final size distribution of Reed-Frost outbreaks is derived in [Lefevre & Picard (1990)](https://www.doi.org/10.2307/1427595) (equation 3.10):

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

In general, the Gontcharoff polynomials are (equation 2.1):

```math
G_k(x | U) = \begin{cases}
  1 & k = 0 \\
  \frac{x^k}{k!} - \sum_{i=0}^{k-1} \frac{u_i^{k-1}}{(k-i)!} G_i(x | U) & k > 0
\end{cases}
```

Define $g(k, q, m) \equiv G_k(1 | q^{m+i}, i \in \mathbb{N})$ so that:

```math
g(k, q, m) = \begin{cases}
  1 & k = 0 \\
  \frac{1}{k!} - \sum_{i=0}^{k-1} q^{(m+i)(k-i)} g(i, q, m) & k > 0
\end{cases}
```

Further define $h(k, q, m) \equiv k! \times g(k, q, m)$ so that:

$$
f(k; n, m, p) = \binom{n}{k} q^{(n-k)(m+k)} h(k, q, m)
$$

and

```math
h(k, q, m) = \begin{cases}
  1 & k = 0 \\
  1 - \sum_{i=0}^{k-1} q^{(m+1)(k-i)} h(i, q, m) & k > 0
\end{cases}
```
