import altair as alt
import numpy as np
import numpy.random
import polars as pl
import streamlit as st

import reedfrost


def app():
    st.set_page_config(
        page_title="Chain binomial models", page_icon="🧮", layout="wide"
    )
    st.title("Chain binomial model")

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
            "Model",
            options=["Reed-Frost", "Enko", "Greenwood"],
            default="Reed-Frost",
        )
        assert model is not None

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

    # derived parameters ------------------------------------------------------
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

    # display initial conditions ----------------------------------------------
    st.subheader("Initial conditions")
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.text(f"Initial susceptible: {n_susceptible}")
    col2.text(f"Initial immune: {n_immune}")
    col3.text(f"Initial infected: {n_infected}")

    # display inputs ---------------------------------------------------------
    st.subheader("Display inputs")
    col1, col2 = st.columns([1, 1])
    metric = col1.segmented_control(
        "Infections metric",
        options=["Cumulative", "Incident"],
        default="Cumulative",
    )
    assert metric is not None

    result_type = col2.segmented_control(
        "Results type",
        options=["Trajectories", "Theoretical"],
        default="Trajectories",
    )
    assert result_type is not None

    st.subheader("Results")
    if result_type == "Trajectories":
        trajectories_chart(
            sim=sim,
            n_simulations=n_simulations,
            metric=metric,
            seed=seed,
        )
    elif result_type == "Theoretical":
        theoretical_chart(
            sim=sim,
            n_susceptible=n_susceptible,
            n_infected=n_infected,
            n_simulations=n_simulations,
            metric=metric,
        )
    else:
        raise ValueError(f"Unknown result type: {result_type}")


def theoretical_chart(
    sim: reedfrost.ChainBinomial,
    n_susceptible: int,
    n_infected: int,
    n_simulations: int,
    metric: str,
    min_bins: int = 10,
    max_bins: int = 20,
    prob_diff_eps: float = 0.005,
    prob_bins: int = 10,
):
    # do the final size pmf ---------------------------------------------------
    # additional no. infected
    k = np.array(range(n_susceptible + 1))
    dens = np.array([sim.prob_final_i_cum_extra(kk) for kk in k])

    final_data = pl.DataFrame(
        {"cum_i_max": k + n_infected, "n_expected": dens * n_simulations}
    ).pipe(
        _bin_data,
        "cum_i_max",
        "n_expected",
        min_bins=min_bins,
        max_bins=max_bins,
    )

    # do the state pmf --------------------------------------------------------

    if metric == "Incident":
        state_data = (
            pl.from_dicts(
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
            )
            .filter(pl.col("t") > 0)
            .pipe(
                _bin_data,
                "Incident",
                "prob",
                max_bins=max_bins,
                min_bins=min_bins,
                group_cols=["t"],
            )
        )

        last_gen = _last_gen_by_prob_change(state_data, "Incident", prob_diff_eps)

        state_chart = (
            alt.Chart(state_data.filter(pl.col("t") <= last_gen))
            .properties(title="Probability of no. of infections by generation")
            .mark_rect()
            .encode(
                alt.X("t:O", title="Generation"),
                alt.Y(
                    "Incident:O",
                    sort=state_data["Incident"].to_list(),
                    title=f"{metric} no. infected",
                ),
                color=alt.condition(
                    alt.datum.prob == 0,
                    alt.value("black"),
                    alt.Color("prob", title="Probability").bin(maxbins=prob_bins),
                ),
            )
        )

        st.altair_chart(state_chart)
    elif metric == "Cumulative":
        state_data = (
            pl.from_dicts(
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
            )
            .filter(pl.col("t") > 0)
            .pipe(
                _bin_data,
                "Cumulative",
                "prob",
                min_bins=min_bins,
                max_bins=max_bins,
                group_cols=["t"],
            )
        )

        last_gen = _last_gen_by_prob_change(state_data, "Cumulative", prob_diff_eps)

        state_chart = (
            alt.Chart(state_data.filter(pl.col("t") <= last_gen))
            .properties(title="Probability of no. of infections by generation")
            .mark_rect()
            .encode(
                alt.X("t:O", title="Generation"),
                alt.Y(
                    "Cumulative:O",
                    sort=state_data["Cumulative"].to_list(),
                    title="Cumulative no. infected",
                ),
                alt.Color("prob", title="Probability").bin(maxbins=prob_bins),
            )
        )

        final_chart = (
            alt.Chart(final_data)
            .properties(title="Total no. of infections")
            .mark_bar()
            .encode(
                alt.Y(
                    "cum_i_max:O",
                    sort=final_data["cum_i_max"].to_list(),
                    title="Cumulative no. infected",
                ),
                alt.X("n_expected"),
            )
        )
        st.altair_chart(state_chart | final_chart)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def trajectories_chart(
    sim: reedfrost.ChainBinomial,
    n_simulations: int,
    seed: int,
    metric: str,
    opacity: float = 0.5,
    stroke_width: float = 1.0,
    jitter: float = 0.1,
):
    # run the simulations ---------------------------------------------------
    rng = numpy.random.default_rng(seed)

    # get one numpy array, representing a timeseries of infections
    # per generation, for each simulation
    simulations = [sim.simulate(rng=child) for child in rng.spawn(n_simulations)]

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

    if metric == "Incident":
        # use just incident infections
        traj_data = traj_data.with_columns(y=pl.col("i"))
    elif metric == "Cumulative":
        # convert to cumulative infections
        traj_data = traj_data.sort(["iter", "t"]).with_columns(
            pl.col("i").cum_sum().over("iter").alias("y")
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # add peak y value by iteration
    traj_data = traj_data.with_columns(
        is_peak=(pl.col("y") == pl.col("y").max()).over("iter")
    )

    # find the maximum y value over all iterations
    max_y = traj_data.select(pl.col("y").max()).item() + 1
    y_axis = alt.Axis(tickCount=max_y + 1)
    y_scale = alt.Scale(domain=[0, max_y])

    # add jitter
    traj_data = traj_data.with_columns(
        y_jitter=pl.col("y")
        + pl.Series("jitter", rng.uniform(-jitter, jitter, traj_data.height))
    )

    line_chart = (
        alt.Chart(traj_data)
        .properties(title="Simulated outbreaks")
        .encode(
            # need +1 because generations are zero-indexed; if last gen is 0, that's
            # one generation
            alt.X("t", title="Generation", axis=alt.Axis(tickCount=last_gen + 1)),
            alt.Y(
                "y_jitter", title=f"{metric} no. infected", axis=y_axis, scale=y_scale
            ),
            alt.Detail("iter"),
        )
        .mark_line(opacity=opacity, strokeWidth=stroke_width)
    )

    hist_chart = (
        alt.Chart(traj_data)
        .transform_calculate(y2=alt.datum.y - 0.5)
        .transform_filter(alt.datum.is_peak)
        .properties(title=f"Maximum {metric} distribution")
        .mark_bar()
        .encode(
            alt.X("count()", title="No. simulations"),
            alt.Y(
                "y2:Q",
                bin=alt.Bin(step=1.0),
                title=f"{metric} no. infected",
                scale=y_scale,
                axis=y_axis,
            ),
        )
    )

    chart = line_chart | hist_chart

    st.altair_chart(chart)


def _bin_data(
    df: pl.DataFrame,
    k_col: str,
    value_col: str,
    min_bins: int,
    max_bins: int,
    group_cols: list[str] = [],
) -> pl.DataFrame:
    k = df[k_col]
    assert min(k) >= 0

    # get the optimal bin size
    bin_size = _get_bin_size(max(k), min_bins=min_bins, max_bins=max_bins)

    # create bins 0-bin_size, bin_size-2*bin_size, etc.
    # the second to last bin includes k
    # we never see the last bin, but we need it up there, so we can see the upper limit
    bin_cuts = np.arange(0, max(k) + bin_size, step=bin_size).tolist()
    bin_cuts.append(bin_cuts[-1] + 1)

    labels = (
        ["<0"]
        + [
            _range_label(bin_cuts[i], bin_cuts[i + 1] - 1)
            for i in range(len(bin_cuts) - 1)
        ]
        + [f">={max(bin_cuts)}"]
    )

    return (
        df.group_by(
            pl.col(k_col)
            .cut(bin_cuts, include_breaks=True, left_closed=True, labels=labels)
            .struct.rename_fields([k_col + "_break", k_col]),
            *group_cols,
        )
        .agg(pl.col(value_col).sum())
        .unnest(k_col)
        .sort(k_col + "_break", descending=True)
    )


def _get_bin_size(k: int, min_bins: int, max_bins: int) -> int:
    """
    For data 0, 1, ..., k, return the size s of bins such that, for a number b
    of bins::

    1. min_bins <= b <= max_bins
    2. s*b >= k
    3. s*b-k is minimized
    """

    if k <= max_bins:
        return 1

    max_error = None
    best_s = None
    for n in range(min_bins, max_bins + 1):
        for s in range(k // n + 1, k + 1):
            error = n * s - k
            if error == 0:
                return s
            elif max_error is None or error < max_error:
                max_error = error
                best_s = s

    if best_s is None:
        raise RuntimeError("No suitable bin size found")

    return best_s


def _range_label(x: int, y: int) -> str:
    if x == y:
        return f"{x}"
    else:
        return f"{x}-{y}"


def _last_gen_by_prob_change(
    df, group: str, eps: float, value: str = "prob", t: str = "t"
) -> int:
    return (
        df.sort([group, t])
        .with_columns(diff=pl.col(value).diff().over(group))
        .filter(pl.col(t) > 0)
        .group_by(t)
        .agg(pl.col("diff").abs().sum())
        .filter(pl.col("diff") > eps)
        .select(pl.col(t).max())
        .item()
    )


if __name__ == "__main__":
    app()
