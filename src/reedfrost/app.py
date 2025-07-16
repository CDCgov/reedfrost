import altair as alt
import numpy as np
import numpy.random
import polars as pl
import polars.datatypes as pdt
import streamlit as st

import reedfrost


def app(opacity=0.5, stroke_width=1.0, jitter=0.1, rect_half_height=0.25, pmf_tol=0.02):
    st.title("Reed-Frost model")

    with st.sidebar:
        st.header("Input parameters")
        n = st.slider("Population size", min_value=1, max_value=100, step=1, value=10)

        # user input is in proportions, but we get the integer number
        n_immune = st.select_slider(
            "Proportion initially immune",
            # values are from 0 to N-1, leaving space for at least 1 infected
            options=range(0, n),
            value=0,
            format_func=lambda x: f"{x / n:.0%}",
        )

        brn = st.slider(
            "Basic reproduction number",
            min_value=0.0,
            max_value=min(15.0, float(n)),
            step=0.1,
            format="%.1f",
            value=min(1.5, float(n)),
        )

        model = st.segmented_control(
            "Model", options=["Reed-Frost", "Greenwood", "Enko"], default="Reed-Frost"
        )
        assert model is not None

        metric = st.segmented_control(
            "Infections metric",
            options=["Cumulative", "Incident"],
            default="Cumulative",
        )
        assert metric is not None

        with st.expander("Advanced options", expanded=False):
            # need special handling for the case where everyone is immune but 1,
            # because streamlit sliders must have a range
            if n - n_immune == 1:
                n_infected = 1
                st.text("No. initially infected: 1")
            else:
                n_infected = st.slider(
                    "No. initially infected",
                    min_value=1,
                    max_value=n - n_immune,
                    step=1,
                    value=1,
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
    n_susceptible = n - n_immune - n_infected
    assert n_susceptible > 0

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

    sim = sim_class(s0=n_susceptible, i0=n_infected, params=params)

    # do the pmf --------------------------------------------------------------
    # additional no. infected
    k = np.array(range(n_susceptible + 1))
    dens = np.array([sim.prob_final_i_cum_extra(kk) for kk in k])

    pmf_data = pl.DataFrame(
        {"cum_i_max": k + n_infected, "n_expected": dens * n_simulations}
    )

    # do the state pmf --------------------------------------------------------
    state_pmf = pl.from_dicts(
        [
            {
                "s": s,
                "Cumulative": n_infected + (n_susceptible - s),
                "t": t,
                "prob": sum(
                    [sim.prob_state(s, i, t) for i in range(n_susceptible + 1)]
                ),
            }
            for s in range(n_susceptible + 1)
            for t in range(n_susceptible + 1)
        ]
    )

    # run the simulations ---------------------------------------------------
    rng = numpy.random.default_rng(seed)

    # get one numpy array, representing a timeseries of infections
    # per generation, for each simulation
    simulations = [sim.simulate(rng=child) for child in rng.spawn(n_simulations)]

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

    # get maximum cumulative infections in each iteration, and put that data only
    # in the first timepoint
    max_i_data = sim_data.group_by("iter").agg(
        pl.col("Cumulative").max().alias("cum_i_max")
    )

    # Combine the different data into a single frame, which helps altair
    # create the common y-axis.
    # Instead of a bar chart, set up rectangles, because altair only does
    # horizontal bar charts with non-quantitative y-axis values, which
    # messes up the common y-axis.
    chart_data = pl.concat(
        [_enforce_schema(df) for df in [sim_data, count_data, pmf_data, max_i_data]],
        how="vertical",
    ).with_columns(
        rect_x=0.0,
        rect_x2=pl.col("n_sims"),
        rect_y=pl.col("cum_i_max") - rect_half_height,
        rect_y2=pl.col("cum_i_max") + rect_half_height,
    )

    # add jitter to avoid overlapping lines
    if jitter > 0:
        chart_data = chart_data.with_columns(
            pl.col("Cumulative", "Incident", "t")
            + pl.Series("jitter", rng.uniform(-jitter, jitter, chart_data.height))
        )

    # common features for multiple charts
    max_y_line = sim_data.select(pl.col(metric).max()).item()
    max_y_pmf = (
        pmf_data.filter(pl.col("n_expected") >= pmf_tol)
        .select(pl.col("cum_i_max").max())
        .item()
    )
    max_y = max(max_y_line, max_y_pmf) + 1
    # common name for cum_i_max
    cum_i_max_title = "Final cumulative no. infected"
    # common scale
    y_scale = alt.Scale(domain=[0, max_y])
    y_axis = alt.Axis(tickCount=last_gen + 1)

    line_chart = (
        alt.Chart(chart_data)
        .properties(title="Simulated outbreaks")
        .encode(
            # need +1 because generations are zero-indexed; if last gen is 0, that's
            # one generation
            alt.X(
                "t",
                title="Generation",
                axis=alt.Axis(tickCount=last_gen + 1),
                scale=alt.Scale(domain=[0, last_gen]),
            ),
            alt.Y(
                metric,
                title=f"{metric} no. infected",
                axis=y_axis,
                scale=y_scale,
            ),
            alt.Detail("iter"),
        )
        .mark_line(opacity=opacity, strokeWidth=stroke_width)
    )

    st.subheader("Initial conditions")
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.text(f"Initial susceptible: {n_susceptible}")
    col2.text(f"Initial immune: {n_immune}")
    col3.text(f"Initial infected: {n_infected}")

    st.subheader("Results")
    match metric:
        case "Incident":
            chart = line_chart
        case "Cumulative":
            hist_chart = (
                alt.Chart(chart_data)
                .properties(title="Final size distribution")
                .mark_rect()
                .encode(
                    alt.X("rect_x", title="No. simulations"),
                    alt.X2("rect_x2"),
                    alt.Y("rect_y", title=cum_i_max_title, scale=y_scale, axis=y_axis),
                    alt.Y2("rect_y2"),
                    tooltip=[
                        alt.Tooltip("cum_i_max", title=cum_i_max_title),
                        alt.Tooltip("n_sims", title="No. simulations"),
                    ],
                )
            )

            pmf_chart = (
                alt.Chart(chart_data)
                .mark_line(
                    color="#ff4b4b",
                    clip=True,
                    point=alt.OverlayMarkDef(color="red", size=50),
                )
                .encode(
                    alt.Y("cum_i_max", scale=y_scale, axis=y_axis),
                    alt.X("n_expected"),
                    # I would have expected to order by cum_i_max, but `iter` works?
                    alt.Order("iter"),
                    tooltip=[
                        alt.Tooltip("cum_i_max", title=cum_i_max_title),
                        alt.Tooltip(
                            "n_expected", title="Expected no. simulations", format=".2f"
                        ),
                    ],
                )
            )

            chart = line_chart | (hist_chart + pmf_chart)

    st.altair_chart(chart)

    state_chart = (
        alt.Chart(state_pmf.filter(pl.col("t") > 0))
        .properties(title="Probability of no. of infections by generation")
        .mark_rect()
        .encode(
            alt.X("t:N", title="Generation"),
            alt.Y("Cumulative:N", sort="descending", title="Cumulative no. infected"),
            color=alt.condition(
                alt.datum.prob == 0,
                alt.value("black"),
                alt.Color("prob", title="Probability").bin(maxbins=10),
            ),
        )
    )

    st.altair_chart(state_chart)


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
