import altair as alt
import numpy as np
import polars as pl
import streamlit as st


def trajectories_chart(
    app,
    opacity: float = 1.0,
    stroke_width: float = 0.5,
    jitter_range: float = 0.8,
    chart_height: float = 500.0,
):
    try:
        new_selection = _parse_selection(app.get_data("selection"))
    except KeyError:
        new_selection = None

    if new_selection is not None:
        app.set_data("y_selected", [new_selection])
    else:
        app.set_data("y_selected", [])

    assert isinstance(app.get_data("traj"), pl.DataFrame)
    assert isinstance(app.get_data("peak_traj"), pl.DataFrame)

    data = (
        app.get_data("traj")
        # add jitter
        .select(["iter", "t", "y"])
        .pipe(_jitter_trajectories, jitter_range=jitter_range)
        # merge the peak & selection data back in
        .join(app.get_data("peak_traj"), on=["iter"], how="left", validate="m:1")
        # put in order for good plotting
        .with_columns(is_selected=(pl.col("peak_y").is_in(app.get_data("y_selected"))))
        .sort("is_selected")
    )

    # find the maximum y value over all iterations
    max_y = data.select(pl.col("y").max()).item()
    last_gen = data.select(pl.col("t").max()).item()

    my_colors = ["#1E4498", "#F78F47"]

    line_chart = (
        alt.Chart(data)
        .properties(title="Simulated outbreaks", height=chart_height)
        .encode(
            # need +1 because generations are zero-indexed; if last gen is 0, that's
            # one generation
            alt.X("t", title="Generation", axis=alt.Axis(tickCount=last_gen + 1)),
            alt.Y(
                "y_jitter",
                title=f"{app.get_data('metric')} no. infected",
                axis=alt.Axis(tickCount=max_y),
                scale=alt.Scale(domain=[0, max_y + 0.5]),
            ),
            alt.Detail("iter"),
            alt.Color(
                "is_selected",
                scale=alt.Scale(range=my_colors),
                legend=None,
            ),
            tooltip=alt.value(None),
        )
        .mark_line(strokeWidth=stroke_width, opacity=opacity)
    )

    hist_data = (
        app.get_data("peak_traj")
        .group_by("peak_y")
        .agg(pl.col("iter").count().alias("count"))
        .join(
            pl.DataFrame({"peak_y": range(max_y + 1), "count": 0}),
            on=["peak_y", "count"],
            how="full",
            coalesce=True,
        )
        .with_columns(is_selected=pl.col("peak_y").is_in(app.get_data("y_selected")))
        .sort("peak_y", descending=True)
    )

    hist_chart = (
        alt.Chart(hist_data)
        .properties(
            title=f"Maximum {app.get_data('metric')} distribution",
            height=chart_height,
        )
        .mark_bar()
        .encode(
            alt.X("count", title="No. simulations"),
            alt.Y(
                "peak_y:N",
                title=f"{app.get_data('metric')} no. infected",
                sort=hist_data["peak_y"].to_list(),
            ),
            alt.Color("is_selected", scale=alt.Scale(range=my_colors), legend=None),
            tooltip=alt.value(None),
        )
        .add_params(
            alt.selection_point("point_selection", on="pointerover", fields=["peak_y"])
        )
    )

    col1, col2 = st.columns([1, 1])
    col1.altair_chart(line_chart)
    col2.altair_chart(hist_chart, on_select="rerun", key="selection")


def theoretical_chart(
    app,
    min_bins: int = 10,
    max_bins: int = 20,
    prob_diff_eps: float = 0.005,
    prob_bins: int = 10,
):
    match app.get_data("metric"):
        case "Incident":
            state_data = app.get_data("state").pipe(
                _bin_data,
                "Incident",
                "prob",
                max_bins=max_bins,
                min_bins=min_bins,
                group_cols=["t"],
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
                        title=f"{app.get_data('metric')} no. infected",
                    ),
                    color=alt.condition(
                        alt.datum.prob == 0,
                        alt.value("black"),
                        alt.Color("prob", title="Probability").bin(maxbins=prob_bins),
                    ),
                )
            )

            st.altair_chart(state_chart)
        case "Cumulative":
            final_data = app.get_data("final").pipe(
                _bin_data,
                "cum_i_max",
                "n_expected",
                min_bins=min_bins,
                max_bins=max_bins,
            )

            state_data = app.get_data("state").pipe(
                _bin_data,
                "Cumulative",
                "prob",
                min_bins=min_bins,
                max_bins=max_bins,
                group_cols=["t"],
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
        case _:
            raise ValueError(f"Unknown metric: {app.get_data('metric')}")


def _parse_selection(x, name="point_selection", value="peak_y") -> int | None:
    assert "selection" in x
    assert name in x["selection"]
    match len(x["selection"][name]):
        case 0:
            return None
        case 1:
            return x["selection"][name][0][value]


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


def _jitter_trajectories(traj: pl.DataFrame, jitter_range=0.8) -> pl.DataFrame:
    assert traj.schema.to_python() == {"iter": int, "t": int, "y": int}
    assert jitter_range >= 0.0

    group_traj = (
        traj.group_by("t", "y")
        .agg(pl.col("iter"))
        .with_columns(n_traj=pl.col("iter").list.len())
    )

    max_counts = (
        group_traj.filter(pl.col("t") > 0).select(pl.col("n_traj").max()).item()
    )
    assert isinstance(max_counts, int)

    space = jitter_range / max_counts

    out = (
        group_traj.with_columns(
            jitter=pl.col("n_traj").map_elements(
                lambda n: _jittered(n, space=space), return_dtype=pl.List(pl.Float64)
            )
        )
        .with_columns(
            y_jitter=pl.col("y")
            + pl.col("jitter") * (pl.when(pl.col("t") > 0).then(1).otherwise(0))
        )
        .select("iter", "t", "y", "y_jitter")
        .explode(["iter", "y_jitter"])
    )

    assert out.shape[0] == traj.shape[0]
    return out


def _jittered(n: int, space: float) -> np.ndarray:
    """Deterministic jitter for n points"""
    half_width = space * (n - 1) / 2
    return np.linspace(-half_width, half_width, num=n)


code = {
    "n_immune_options": lambda app: list(range(0, app.get_data("n"))),
    "n_immune_format_func": lambda app: lambda x: f"{x / app.get_data('n'):.0%}",
    "brn_max_value": lambda app: min(15.0, float(app.get_data("n"))),
    "n_infected_max_value": lambda app: app.get_data("n") - app.get_data("n_immune"),
    "seed_max_value": lambda _: 2**32 - 1,
    "get_n_susceptible": lambda app: app.get_data("n_susceptible"),
    "get_n_immune": lambda app: app.get_data("n_immune"),
    "get_n_infected": lambda app: app.get_data("n_infected"),
    "one_susceptible": lambda app: app.get_data("n") - app.get_data("n_immune") == 1,
    "results_not_available": lambda app: not app.get_data("results"),
    "results_are_trajectories": lambda app: app.get_data("result_type")
    == "Trajectories",
    "trajectories_chart": lambda app: trajectories_chart(app),
    "theoretical_chart": lambda app: theoretical_chart(app),
}
