import numpy as np
import numpy.random
import polars as pl
import streamlit as st

import reedfrost


def run_model(app):
    app.set_data("results", False)

    # derive parameters
    result_type = app.get_data("result_type")
    metric = app.get_data("metric")

    app.set_data(
        "n_susceptible",
        app.get_data("n") - app.get_data("n_immune") - app.get_data("n_infected"),
    )

    # n changed "quickly" and the defaults did not reset on the other two first
    if app.get_data("n_susceptible") < 0:
        app.set_data(
            "n_susceptible",
            app.get_data("n")
            - app.get_default("n_immune")
            - app.get_default("n_infected"),
        )

    assert app.get_data("n_susceptible") > 0

    match (result_type, metric):
        case ("Trajectories", _):
            model_trajectories(app)
        case ("Theoretical", "Incident"):
            model_theoretical_incident(app)
        case ("Theoretical", "Cumulative"):
            model_theoretical_cumulative(app)
        case _:
            raise ValueError(f"Unknown results/metric: {result_type}/{metric}")

    app.set_data("results", True)


def model_trajectories(app):
    params = sim_params_from_app(app)
    sim = _build_sim(params)
    rng = numpy.random.default_rng(app.get_data("seed"))

    # get one numpy array, representing a timeseries of infections
    # per generation, for each simulation
    simulations = [
        sim.simulate(rng=child) for child in rng.spawn(app.get_data("n_simulations"))
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

    metric = app.get_data("metric")
    match metric:
        case "Incident":
            # use just incident infections
            traj_data = traj_data.with_columns(pl.col("i").alias("y"))
        case "Cumulative":
            # convert to cumulative infections
            traj_data = traj_data.sort(["iter", "t"]).with_columns(
                pl.col("i").cum_sum().over("iter").alias("y")
            )
        case _:
            raise ValueError(f"Unknown metric: {metric}")

    # get peak value by iteration
    peak_traj_data = traj_data.group_by("iter").agg(pl.col("y").max().alias("peak_y"))

    app.set_data("traj", traj_data)
    app.set_data("peak_traj", peak_traj_data)


def model_theoretical_cumulative(app):
    assert app.get_data("result_type") == "Theoretical"
    assert app.get_data("metric") == "Cumulative"

    params = sim_params_from_app(app)
    sim = _build_sim(params)
    n_susceptible = app.get_data("n_susceptible")
    n_infected = app.get_data("n_infected")
    n_simulations = app.get_data("n_simulations")

    # do the final size pmf ---------------------------------------------------
    # additional no. infected
    k = np.array(range(n_susceptible + 1))
    dens = np.array([sim.prob_final_i_cum_extra(kk) for kk in k])

    final_data = pl.DataFrame(
        {
            "cum_i_max": k + n_infected,
            "n_expected": dens * n_simulations,
        }
    )

    state_data = pl.from_dicts(
        [
            {
                "Cumulative": n_infected + (n_susceptible - s),
                "t": t,
                "prob": sum(
                    [sim.prob_state(s, i, t) for i in range(n_susceptible + 1)]
                ),
            }
            for s in range(n_susceptible + 1)
            for t in range(n_susceptible + 1)
        ]
    ).filter(pl.col("t") > 0)

    app.set_data("final", final_data)
    app.set_data("state", state_data)


def model_theoretical_incident(app):
    assert app.get_data("result_type") == "Theoretical"
    assert app.get_data("metric") == "Incident"

    params = sim_params_from_app(app)
    sim = _build_sim(params)
    n_susceptible = app.get_data("n_susceptible")

    state_data = pl.from_dicts(
        [
            {
                "Incident": i,
                "t": t,
                "prob": sum(
                    [sim.prob_state(s, i, t) for s in range(n_susceptible + 1)]
                ),
            }
            for i in range(n_susceptible + 1)
            for t in range(n_susceptible + 1)
        ]
    ).filter(pl.col("t") > 0)

    app.set_data("state", state_data)


def sim_params_from_app(app):
    out = {}
    out["model"] = app.get_data("model")
    out["n_susceptible"] = app.get_data("n_susceptible")
    out["n_infected"] = app.get_data("n_infected")
    out["brn"] = app.get_data("brn")
    out["n"] = app.get_data("n")
    return out


def _build_sim(p: dict) -> reedfrost.ChainBinomial:
    match p["model"]:
        case "Reed-Frost":
            params = {"p": p["brn"] / p["n"]}
            sim_class = reedfrost.ReedFrost
        case "Greenwood":
            params = {"p": p["brn"] / p["n"]}
            sim_class = reedfrost.Greenwood
        case "Enko":
            params = {
                "n": p["n"],
                "k": np.log(1.0 - p["brn"] / p["n"])
                / np.log(1.0 - 1.0 / (p["n"] - 1.0)),
            }
            sim_class = reedfrost.Enko
        case _:
            raise ValueError(f"Unknown model: {p['model']}")

    return sim_class(s0=p["n_susceptible"], i0=p["n_infected"], params=params)


if st.runtime.exists():
    _build_sim = st.cache_resource(_build_sim)
