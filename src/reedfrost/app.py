import altair as alt
import numpy as np
import numpy.random
import polars as pl
import streamlit as st

import reedfrost as rf


def app():
    st.title("Reed-Frost model")

    with st.sidebar:
        st.header("Input parameters")
        n_susceptible = st.slider(
            "No. initially susceptible", min_value=1, max_value=50, step=1, value=10
        )
        n_infected = st.slider(
            "No. initially infected", min_value=1, max_value=10, step=1, value=1
        )
        reff = st.slider(
            "Effective reproduction number",
            min_value=0.0,
            max_value=min(5.0, float(n_susceptible)),
            step=0.01,
            format="%.2f",
            value=min(1.5, float(n_susceptible)),
        )

        if n_susceptible == 0:
            p = 0.0
        else:
            p = reff / n_susceptible

        st.header("Stochastic parameters")

        n_simulations = st.slider(
            "No. simulations",
            min_value=5,
            max_value=250,
            step=1,
            value=50,
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**32 - 1,
            step=1,
            value=42,
        )

        metric = st.segmented_control(
            "Metric", options=["Cumulative", "Incident"], default="Cumulative"
        )
        assert metric is not None

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    simulation_chart = simulations(
        n_susceptible=n_susceptible,
        n_infected=n_infected,
        p=p,
        metric=metric,
        n_simulations=n_simulations,
        seed=seed,
    )

    match metric:
        case "Incident":
            st.altair_chart(simulation_chart)
        case "Cumulative":
            pmf_chart = final_size_distribution(
                n_susceptible=n_susceptible,
                n_infected=n_infected,
                p=p,
            )

            st.altair_chart(simulation_chart | pmf_chart)


def final_size_distribution(n_susceptible: int, n_infected: int, p: float) -> alt.Chart:
    """Calculate and display the final size distribution"""
    # additional no. infected
    k = np.array(range(n_susceptible + 1))
    dens = rf.pmf(k=k, s=n_susceptible, i=n_infected, p=p)

    data = pl.concat(
        [
            pl.DataFrame({"total_infected": k + n_infected, "dens": dens * 100}),
            pl.DataFrame({"total_infected": range(n_infected), "dens": 0.0}),
        ]
    )

    return (
        alt.Chart(data)
        .properties(title="Final size distribution")
        .encode(
            alt.Y("total_infected:N", title="Total no. infected", sort="descending"),
            alt.X("dens", title="Probability (%)"),
            tooltip=[
                alt.Tooltip("total_infected:N", title="Total no. infected"),
                alt.Tooltip("dens:Q", format=".1f", title="Probability (%)"),
            ],
        )
        .mark_bar()
    )


def simulations(
    n_susceptible: int,
    n_infected: int,
    p: float,
    n_simulations: int,
    metric: str,
    seed: int,
    jitter: float = 0.1,
    opacity: float = 0.75,
    stroke_width: float = 1.0,
) -> alt.Chart:
    """Run and display stochastic simulations"""

    assert metric in ["Incident", "Cumulative"]

    rng = numpy.random.default_rng(seed)

    # get one numpy array, representing a timeseries of infections
    # per generation, for each simulation
    results = [
        rf.simulate(s=n_susceptible, i=n_infected, p=p, rng=child)
        for child in rng.spawn(n_simulations)
    ]

    # combine into a dataframe
    data = (
        pl.concat(
            [
                pl.DataFrame({"iter": k, "t": range(len(res)), "Incident": res})
                for k, res in enumerate(results)
            ]
        )
        .sort(["iter", "t"])
        # conver to cumulative infections
        .with_columns(pl.col("Incident").cum_sum().over("iter").alias("Cumulative"))
    )

    # remove entries where no infections occurred
    last_gen = data.filter(pl.col("Incident") > 0).select(pl.col("t").max()).item()
    data = data.filter(pl.col("t") <= last_gen)

    # keep track of y-axis limit
    max_y = data.select(pl.col(metric).max()).item() + 1

    # add jitter to the y-axis to avoid overlapping lines
    if jitter > 0:
        data = data.with_columns(
            pl.col(metric)
            + pl.Series("jitter", np.random.uniform(-jitter, jitter, data.shape[0]))
        )

    return (
        alt.Chart(data)
        .properties(title="Stochastic simulations")
        .encode(
            alt.X("t", title="Generation", axis=alt.Axis(tickCount=last_gen + 1)),
            alt.Y(
                metric, title=f"{metric} no. infected", axis=alt.Axis(tickCount=max_y)
            ),
            alt.Detail("iter"),
        )
        .mark_line(opacity=opacity, strokeWidth=stroke_width)
    )


if __name__ == "__main__":
    app()
