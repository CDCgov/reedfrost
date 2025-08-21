import numpy as np
import numpy.random
import polars as pl
import streamlit as st

import reedfrost


def get_results(params) -> dict:
    # derive parameters
    n_susceptible = params["n"] - params["n_immune"] - params["n_infected"]
    assert n_susceptible > 0
    params["n_susceptible"] = n_susceptible

    match (params["result_type"], params["metric"]):
        case ("Trajectories", _):
            return model_trajectories(params)
        case ("Theoretical", "Incident"):
            return model_theoretical_incident(params)
        case ("Theoretical", "Cumulative"):
            return model_theoretical_cumulative(params)
        case _:
            raise ValueError(
                f"Unknown results/metric: {params['result_type']}/{params['metric']}"
            )


def model_trajectories(params: dict) -> dict:
    sim = _build_sim(**params)
    rng = numpy.random.default_rng(params["seed"])

    # get one numpy array, representing a timeseries of infections
    # per generation, for each simulation
    simulations = [
        sim.simulate(rng=child) for child in rng.spawn(params["n_simulations"])
    ]

    # combine those simulations into a dataframe, making trajectories
    traj_data = pl.concat(
        [
            pl.DataFrame({"iter": k, "t": range(len(x)), "i": x})
            for k, x in enumerate(simulations)
        ]
    )

    # remove entries where no infections occurred
    last_gen = traj_data.filter(pl.col("i") > 0).select(pl.col("t").max()).item()
    traj_data = traj_data.filter(pl.col("t") <= last_gen)

    match params["metric"]:
        case "Incident":
            # use just incident infections
            traj_data = traj_data.with_columns(pl.col("i").alias("y"))
        case "Cumulative":
            # convert to cumulative infections
            traj_data = traj_data.sort(["iter", "t"]).with_columns(
                pl.col("i").cum_sum().over("iter").alias("y")
            )
        case _:
            raise ValueError(f"Unknown metric: {params['metric']}")

    # get peak value by iteration
    peak_traj_data = traj_data.group_by("iter").agg(pl.col("y").max().alias("peak_y"))

    return {"traj": traj_data, "peak_traj": peak_traj_data}


def model_theoretical_cumulative(params: dict) -> dict:
    assert params["result_type"] == "Theoretical"
    assert params["metric"] == "Cumulative"

    sim = _build_sim(**params)

    # do the final size pmf ---------------------------------------------------
    # additional no. infected
    k = np.array(range(params["n_susceptible"] + 1))
    dens = np.array([sim.prob_final_i_cum_extra(kk) for kk in k])

    final_data = pl.DataFrame(
        {
            "cum_i_max": k + params["n_infected"],
            "n_expected": dens * params["n_simulations"],
        }
    )

    state_data = pl.from_dicts(
        [
            {
                "Cumulative": params["n_infected"] + (params["n_susceptible"] - s),
                "t": t,
                "prob": sum(
                    [
                        sim.prob_state(s, i, t)
                        for i in range(params["n_susceptible"] + 1)
                    ]
                ),
            }
            for s in range(params["n_susceptible"] + 1)
            for t in range(params["n_susceptible"] + 1)
        ]
    ).filter(pl.col("t") > 0)

    return {"final": final_data, "state": state_data}


def model_theoretical_incident(params: dict) -> dict:
    assert params["result_type"] == "Theoretical"
    assert params["metric"] == "Incident"

    sim = _build_sim(**params)

    state_data = pl.from_dicts(
        [
            {
                "Incident": i,
                "t": t,
                "prob": sum(
                    [
                        sim.prob_state(s, i, t)
                        for s in range(params["n_susceptible"] + 1)
                    ]
                ),
            }
            for i in range(params["n_susceptible"] + 1)
            for t in range(params["n_susceptible"] + 1)
        ]
    ).filter(pl.col("t") > 0)

    return {"state": state_data}


@st.cache_resource
def _build_sim(
    model: str,
    n_susceptible: int,
    n_infected: int,
    brn: float,
    n: int,
    **kwargs,
) -> reedfrost.ChainBinomial:
    match model:
        case "Reed-Frost":
            params = {"p": brn / n}
            sim_class = reedfrost.ReedFrost
        case "Greenwood":
            params = {"p": brn / n}
            sim_class = reedfrost.Greenwood
        case "Enko":
            params = {
                "n": n,
                "k": np.log(1.0 - brn / n) / np.log(1.0 - 1.0 / (n - 1.0)),
            }
            sim_class = reedfrost.Enko
        case _:
            raise ValueError(f"Unknown model: {model}")

    return sim_class(s0=n_susceptible, i0=n_infected, params=params)
