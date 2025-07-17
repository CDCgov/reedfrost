# Model descriptions

Chain binomial models are discrete time, discrete person, two-dimensional Markov chain models. Epidemiologically, these models track numbers of susceptibles $S_t$ and infected $I_t$ in each generation $t$.

In probability theoretic notation, we write $S_t$ and $I_t$ to mean the events that there are a certain number of susceptible and infected individuals in generation $t$. Given an initial state $(S_0, I_0)$ and a transition probability $P(S_t, I_t | S_{t-1}, I_{t-1})$, the probability of a particular trajectory $(S_0, I_0, S_1, I_1, \ldots, S_t, I_t)$ is:

```math
P(S_t, I_t | S_{t-1}, I_{t-1}) \ldots P(S_1, I_1 | S_0, I_0)
```

## General assumptions and corollaries

- Use an susceptible-infectious-removed infection course, with a fixed population size $n$, so that there always an implicit removed population $n - S_t - I_t$.
- Infections last one generation.
- $I_t \sim \mathrm{Binom}(S_t, \pi(I_{t-1}))$, where $\pi$ is a function that determines the binomial probability based on the number of prior infections (and some fixed parameters). We require that $\pi(0)=0$. Note that the basic reproduction number is $n \cdot \pi(1)$.
- $S_t = S_{t-1} - I_t$. The number of susceptibles is non-increasing.
- There is some stopping time $\tau$ such that $I_t=0$ and $S_t = S_\tau$ for all $t \geq \tau$.

## Models

### Reed-Frost

Each infected person has an independent probability $p$ of infecting each susceptible person:

```math
\pi(i) = 1 - (1 - p)^i
```

The basic reproduction number is $R_0 = np$.

### Greenwood

If there is at least one infected person, then each susceptible person has a static probability $p$ of being infected:

```math
\pi(i) = \begin{cases}
p & i > 0 \\
0 & i = 0
\end{cases}
```

Like for the Reed-Frost model, $R_0=np$. Thus, if a Reed-Frost and Greenwood simulation are matched on $R_0$, then the Greenwood model will generally proceed with a lower force of infection.

### En'ko

Each susceptible person makes $k$ successful contacts per generation:

```math
\pi(i) = 1 - \left(1 - \frac{i}{n-1}\right)^k
```

In this case, $R_0 = n \left[ 1 - \left(1 - \tfrac{1}{n-1}\right)^k \right]$, from which we can derive:

```math
k = \frac{\log (1 - R_0/n)}{\log [1 - 1/(n-1)]}
```

## Implementation

Let $P(s, i, t)$ be the probability of being in state $(s, i)$ in generation $t$. Begin from an initial state, setting $P(s_0, i_0, 0) = 1$. Then iteratively generate:

```math
P(s, i, t) = \sum_{i'=0}^{s_0-s-i} f_\mathrm{Binom}(i; s+i, \pi(i') ) \cdot P(s + i, i', t - 1)
```

These values can then be summarized to generate outcomes like final size distributions and the probability of intermediate states.

This implementation has $\mathcal{O}(n^3)$ complexity, which in practice is not limiting for simulations with $n \lesssim 100$.

### Alternative implementations

[Lefevre & Picard (1990)](https://www.doi.org/10.2307/1427595) describe a $\mathcal{O}(n)$ algorithm for computing the final size distribution of Reed-Frost outbreaks. In practice, we found that this approach suffers from problems with overflows and numerical instability that were less tractable than the memory and time challenges with the dynamic programming approach. This approach also does not permit the use of different chain binomial models or the intermediate state probability calculations.

[Barbour & Utev (2004)](https://ideas.repec.org/a/eee/spapps/v113y2004i2p173-197.html) describe a $\mathcal{O}(n)$ algorithm for computing an approximation of the final size distribution that is valid for large Reed-Frost outbreaks. This approach is tractable and numerically stable, but it is limited to large outbreaks, does not permit the use of different chain binomial models or the intermediate state probability calculations.
