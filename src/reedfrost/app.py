import altair as alt
import numpy as np
import numpy.random
import polars as pl
import polars.datatypes as pdt
import streamlit as st

import reedfrost as rf


def app(opacity=0.5, stroke_width=1.0, jitter=0.1):
    st.title("Reed-Frost model")

    with st.sidebar:
        st.header("Input parameters")
        n_susceptible = st.slider(
            "No. initially susceptible", min_value=1, max_value=50, step=1, value=10
        )
        reff = st.slider(
            "Effective reproduction number",
            min_value=0.0,
            max_value=min(5.0, float(n_susceptible)),
            step=0.1,
            format="%.1f",
            value=min(1.5, float(n_susceptible)),
        )

        metric = st.segmented_control(
            "Infections metric",
            options=["Cumulative", "Incident"],
            default="Cumulative",
        )
        assert metric is not None

        with st.expander("Advanced options", expanded=False):
            n_infected = st.slider(
                "No. initially infected", min_value=1, max_value=10, step=1, value=1
            )

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

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    # derived parameters
    if n_susceptible == 0:
        p = 0.0
    else:
        p = reff / n_susceptible

    # do the pmf --------------------------------------------------------------
    # additional no. infected
    k = np.array(range(n_susceptible + 1))
    dens = rf.pmf(k=k, s=n_susceptible, i=n_infected, p=p)

    pmf_data = pl.DataFrame(
        {"cum_i_max": k + n_infected, "n_expected": dens * n_simulations}
    )

    # run the simulations ---------------------------------------------------
    rng = numpy.random.default_rng(seed)

    # get one numpy array, representing a timeseries of infections
    # per generation, for each simulation
    simulations = [
        rf.simulate(s=n_susceptible, i=n_infected, p=p, rng=child)
        for child in rng.spawn(n_simulations)
    ]

    # combine into a dataframe
    sim_data = (
        pl.concat(
            [
                pl.DataFrame({"iter": k, "t": range(len(x)), "Incident": x})
                for k, x in enumerate(simulations)
            ]
        )
        .sort(["iter", "t"])
        # convert to cumulative infections
        .with_columns(pl.col("Incident").cum_sum().over("iter").alias("Cumulative"))
    )

    # remove entries where no infections occurred
    last_gen = sim_data.filter(pl.col("Incident") > 0).select(pl.col("t").max()).item()
    sim_data = sim_data.filter(pl.col("t") <= last_gen)

    count_data = (
        sim_data.group_by("iter")
        .agg(pl.col("Cumulative").max().alias("cum_i_max"))
        .group_by("cum_i_max")
        .agg(pl.count().alias("n_sims"))
    )

    # ensure that count data have all the possible values
    count_data = (
        pl.DataFrame(
            {"cum_i_max": list(range(n_infected, n_infected + n_susceptible + 1))}
        )
        .join(count_data, on="cum_i_max", how="left")
        .with_columns(pl.col("n_sims").fill_null(0))
    )

    max_y = sim_data.select(pl.col(metric).max()).item() + 1

    # get maximum cumulative infections in each iteration, and put that data only
    # in the first timepoint
    max_i_data = sim_data.group_by("iter").agg(
        pl.col("Cumulative").max().alias("cum_i_max")
    )

    chart_data = pl.concat(
        [_enforce_schema(df) for df in [sim_data, count_data, pmf_data, max_i_data]],
        how="vertical",
    )

    # add jitter to avoid overlapping lines
    if jitter > 0:
        chart_data = chart_data.with_columns(
            pl.col("Cumulative", "Incident", "t")
            + pl.Series("jitter", np.random.uniform(-jitter, jitter, chart_data.height))
        )

    line_chart = (
        alt.Chart(chart_data)
        .properties(title="Stochastic simulations")
        .encode(
            alt.X("t", title="Generation", axis=alt.Axis(tickCount=last_gen + 1)),
            alt.Y(
                metric,
                title=f"{metric} no. infected",
                scale=alt.Scale(domain=[0, max_y]),
            ),
            alt.Detail("iter"),
        )
        .mark_line(opacity=opacity, strokeWidth=stroke_width)
    )

    # common name for cum_i_max
    cum_i_max_title = "Final cumulative no. infected"

    # common tooltip for layered hist+pmf chart
    tooltip = [
        alt.Tooltip("cum_i_max", title=cum_i_max_title),
        alt.Tooltip("n_sims", title="No. simulations"),
        alt.Tooltip("n_expected", title="Expected no. simulations", format=".1f"),
    ]

    scale = alt.Scale(domain=[0, max_y])

    match metric:
        case "Incident":
            chart = line_chart
        case "Cumulative":
            hist_chart = (
                alt.Chart(chart_data)
                .mark_point()
                .encode(
                    alt.Y("cum_i_max", title=cum_i_max_title, scale=scale),
                    alt.X("n_sims", title="No. simulations"),
                    tooltip=tooltip,
                )
            )

            pmf_chart = (
                alt.Chart(chart_data)
                .mark_point(color="#ff4b4b")
                .encode(
                    alt.Y("cum_i_max", scale=scale),
                    alt.X("n_expected"),
                    tooltip=tooltip,
                )
            )

            chart = line_chart | (hist_chart + pmf_chart)

    st.altair_chart(chart)


def _enforce_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure data frame has the expected columns, adding null columns as needed"""
    schema = [
        ("iter", pdt.Int64),
        ("t", pdt.Int64),
        ("Incident", pdt.Int64),
        ("Cumulative", pdt.Int64),
        ("cum_i_max", pdt.Int64),
        ("n_sims", pdt.Int64),
        ("n_expected", pdt.Float64),
    ]

    schema_cols = [x[0] for x in schema]
    assert set(df.columns).issubset(schema_cols)
    new_cols = set(schema_cols) - set(df.columns)

    return df.select(
        *[
            pl.lit(None).alias(name).cast(type_)
            if name in new_cols
            else pl.col(name).cast(type_)
            for name, type_ in schema
        ]
    ).select(schema_cols)


if __name__ == "__main__":
    app()
