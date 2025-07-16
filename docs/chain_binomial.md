# Chain binomial models

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
