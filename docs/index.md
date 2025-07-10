# The Reed-Frost model

In [the Reed-Frost model](https://en.wikipedia.org/wiki/Reed%E2%80%93Frost_model), there are susceptible, infected, and removed (recovered or immune) people. In each generation, each infected person has an independent probability $p$ to infect each susceptible person. Infections last for one generation, so infected people are removed after they have one chance to infect each susceptible person.

Let $S_t$, $I_t$, and $R_t$ represent the 3 compartments. An epidemic progresses according to:

$$
\begin{align*}
I_{t+1} &\sim \mathrm{Binomial}\left[S_t; 1-(1-p)^{I_t}\right] \\
S_{t+1} &= S_t - I_{t+1} \\
R_{t+1} &= R_t + I_t
\end{align*}
$$

for $t > 0$.

The basic reproduction number is $\mathcal{R}_0=pN$, where $N = S_0 + I_0 + R_0$.

## Final sizes

The `reedfrost` package and app calculate the distribution of the final sizes of outbreaks, that is, the total number of people infected beyond the initial infections.
