# The Reed-Frost model

In [the model](https://en.wikipedia.org/wiki/Reed%E2%80%93Frost_model), there are susceptible and infected people. In each generation, each infected person has an independent probability $p$ to infect each susceptible person. Infections last for one generation, so infected people are removed after they have one chance to infect each susceptible person.

Let $S_0$ and $I_0$ be the initial numbers susceptible and infected. Then the number infected and susceptible in each generation $t \geq 0$ is:

$$
\begin{align*}
I_{t+1} &\sim \mathrm{Binomial}\left[S_t; 1-(1-p)^{I_t}\right] \\
S_{t+1} &= S_t - I_{t+1}
\end{align*}
$$

This repo calculates the distribution of the final sizes of outbreaks, that is, the total number of people infected beyond the initial infections.
