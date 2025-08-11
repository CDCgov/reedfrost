import importlib.resources
import json
from types import ModuleType
from typing import Any, Callable, Literal

import altair as alt
import numpy as np
import numpy.random
import polars as pl
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import reedfrost


class Inputter:
    def __init__(self):
        """
        Track streamlit input components and the values they return
        """
        self.inputs = {}

    def add_component(
        self,
        key: str,
        method_name: str,
        kwargs: dict,
        kwargs_type: Literal["fixed", "callable"] = "callable",
        c: DeltaGenerator | ModuleType | None = None,
    ):
        """
        Add a streamlit input component. The resulting value is stored in
        self.inputs

        In the simplest case, the function arguments to the component method are fixed
        (e.g., a slider that goes from 0 to 100). In other cases, a component depends on
        inputs collected before it (e.g., a slider that goes from 0 to N, where N was
        set by a previous slider).

        If `kwargs_type` is `"callable"`, and a value in `kwargs` is callable, then that
        value will be replaced by the result of calling that callable on the dictionary
        of inputs collected so far. If `"fixed"`, then the values are not updated.
        If you want a kwarg to be a function (e.g., a label formatter), then either use
        `"fixed"` or allow for a dummy call like `lambda _: lambda x: my_formatting(x)`.

        Args:
            key: The key under which the value will be stored in self.inputs
            method_name: The name of the streamlit method to call (e.g., "slider")
            kwargs: The keyword arguments to pass to the method (e.g., "label", "min_value")
            kwargs_type: how to update the kwargs before calling the method
            c: The streamlit component or module to use; if None, use st
        """
        if c is None:
            c = st

        assert method_name in dir(c)
        assert key not in self.inputs

        match kwargs_type:
            case "fixed":
                pass
            case "callable":
                kwargs = {
                    k: v(self.inputs) if callable(v) else v for k, v in kwargs.items()
                }
            case _:
                raise RuntimeError(f"Unknown kwargs_type: {kwargs_type}")

        self.inputs[key] = getattr(c, method_name)(**kwargs)

    def add_custom_component(self, key: str, method: Callable[[dict[str, Any]], Any]):
        assert key not in self.inputs
        self.inputs[key] = method(self.inputs)

    def add_value(self, key: str, value):
        assert key not in self.inputs
        self.inputs[key] = value


def app():
    st.set_page_config(
        page_title="Chain binomial models", page_icon="🧮", layout="wide"
    )
    st.title("Chain binomial models")

    with st.sidebar:
        params = get_params()

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    results = get_results(params)
    view(params, results)


def get_params() -> dict:
    st.header("Input parameters")
    inputter = Inputter()
    inputter.add_component(
        "n",
        "slider",
        kwargs={
            "label": "Population size",
            "min_value": 1,
            "max_value": 100,
            "step": 1,
            "value": 10,
        },
    )

    # user input is in proportions, but we get the integer number
    inputter.add_component(
        "n_immune",
        "select_slider",
        kwargs={
            "label": "Proportion initially immune",
            # values are from 0 to N-1, leaving space for at least 1 infected
            "options": lambda p: range(0, p["n"]),
            "value": 0,
            "format_func": lambda p: lambda x: f"{x / p['n']:.0%}",
        },
    )

    inputter.add_component(
        "brn",
        "slider",
        kwargs={
            "label": "Basic reproduction number",
            "min_value": 0.0,
            "max_value": lambda p: min(15.0, float(p["n"])),
            "step": 0.1,
            "format": "%.1f",
            "value": lambda p: min(1.5, float(p["n"])),
        },
    )

    inputter.add_component(
        "model",
        "segmented_control",
        kwargs={
            "label": "Model",
            "options": ["Reed-Frost", "Enko", "Greenwood"],
            "default": "Reed-Frost",
        },
    )

    inputter.add_component(
        "result_type",
        "segmented_control",
        kwargs={
            "label": "Results type",
            "options": ["Trajectories", "Theoretical"],
            "default": "Trajectories",
        },
    )

    inputter.add_component(
        "metric",
        "segmented_control",
        kwargs={
            "label": "Infections metric",
            "options": ["Cumulative", "Incident"],
            "default": "Cumulative",
        },
    )

    with st.expander("Advanced options", expanded=False):
        # need special handling for the case where everyone is immune but 1,
        # because streamlit sliders must have a range
        inputter.add_custom_component("n_infected", n_infected_component)

        inputter.add_component(
            "n_simulations",
            "slider",
            kwargs={
                "label": "No. simulations",
                "min_value": 5,
                "max_value": 250,
                "step": 1,
                "value": 100,
            },
        )

        inputter.add_component(
            "seed",
            "number_input",
            kwargs={
                "label": "Random seed",
                "min_value": 0,
                "max_value": 2**32 - 1,
                "step": 1,
                "value": 42,
            },
        )

    return inputter.inputs


def n_infected_component(params: dict) -> int:
    if params["n"] - params["n_immune"] == 1:
        st.text("No. initially infected: 1")
        return 1
    else:
        return st.slider(
            "No. initially infected",
            min_value=1,
            max_value=params["n"] - params["n_immune"],
            step=1,
            value=1,
        )


@st.cache_resource
def load_inputs() -> dict:
    with importlib.resources.open_text("reedfrost", "inputs.json") as f:
        inputs = json.load(f)

    return inputs


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


def view(params: dict, results: dict) -> None:
    # display initial conditions ----------------------------------------------
    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    col1.metric("Initial susceptible", params["n_susceptible"])
    col2.metric("Initial immune", params["n_immune"])
    col3.metric("Initial infected", params["n_infected"])

    # results -----------------------------------------------------------------
    view_c = st.empty()
    view_c.text("Calculating...")

    if results is None:
        pass
    else:
        match params["result_type"]:
            case "Trajectories":
                trajectories_chart(c=view_c, results=results, params=params)
            case "Theoretical":
                theoretical_chart(c=view_c, results=results, params=params)
            case _:
                raise ValueError(f"Unknown result type: {params['result_type']}")


def theoretical_chart(
    c: DeltaGenerator,
    results: dict,
    params: dict,
    min_bins: int = 10,
    max_bins: int = 20,
    prob_diff_eps: float = 0.005,
    prob_bins: int = 10,
):
    match params["metric"]:
        case "Incident":
            state_data = results["state"].pipe(
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
                        title=f"{params['metric']} no. infected",
                    ),
                    color=alt.condition(
                        alt.datum.prob == 0,
                        alt.value("black"),
                        alt.Color("prob", title="Probability").bin(maxbins=prob_bins),
                    ),
                )
            )

            c.altair_chart(state_chart)
        case "Cumulative":
            final_data = results["final"].pipe(
                _bin_data,
                "cum_i_max",
                "n_expected",
                min_bins=min_bins,
                max_bins=max_bins,
            )

            state_data = results["state"].pipe(
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
            c.altair_chart(state_chart | final_chart)
        case _:
            raise ValueError(f"Unknown metric: {params['metric']}")


def trajectories_chart(
    c: DeltaGenerator,
    params: dict,
    results: dict,
    opacity: float = 1.0,
    stroke_width: float = 0.5,
    jitter_range: float = 0.8,
    chart_height: float = 500.0,
):
    if "y_selected" not in st.session_state:
        # initialize the y selection
        st.session_state["y_selected"] = []
    elif "selection" in st.session_state:
        # respond to the selection, if any
        new_selection = _parse_selection(st.session_state["selection"])
        if new_selection is not None:
            st.session_state["y_selected"] = [new_selection]
        else:
            st.session_state["y_selected"] = []

    assert isinstance(results["traj"], pl.DataFrame)
    assert isinstance(results["peak_traj"], pl.DataFrame)

    data = (
        results["traj"]
        # add jitter
        .select(["iter", "t", "y"])
        .pipe(_jitter_trajectories, jitter_range=jitter_range)
        # merge the peak & selection data back in
        .join(results["peak_traj"], on=["iter"], how="left", validate="m:1")
        # put in order for good plotting
        .with_columns(
            is_selected=(
                pl.col("peak_y").is_in(st.session_state.y_selected)
                if "y_selected" in st.session_state
                else pl.lit(False)
            )
        )
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
                title=f"{params['metric']} no. infected",
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
        results["peak_traj"]
        .group_by("peak_y")
        .agg(pl.col("iter").count().alias("count"))
        .join(
            pl.DataFrame({"peak_y": range(max_y + 1), "count": 0}),
            on=["peak_y", "count"],
            how="full",
            coalesce=True,
        )
        .with_columns(is_selected=pl.col("peak_y").is_in(st.session_state.y_selected))
        .sort("peak_y", descending=True)
    )

    hist_chart = (
        alt.Chart(hist_data)
        .properties(
            title=f"Maximum {params['metric']} distribution", height=chart_height
        )
        .mark_bar()
        .encode(
            alt.X("count", title="No. simulations"),
            alt.Y(
                "peak_y:N",
                title=f"{params['metric']} no. infected",
                sort=hist_data["peak_y"].to_list(),
            ),
            alt.Color("is_selected", scale=alt.Scale(range=my_colors), legend=None),
            tooltip=alt.value(None),
        )
        .add_params(
            alt.selection_point("point_selection", on="pointerover", fields=["peak_y"])
        )
    )

    col1, col2 = c.columns([1, 1])
    col1.altair_chart(line_chart)
    col2.altair_chart(hist_chart, on_select="rerun", key="selection")


def _parse_selection(x, name="point_selection", value="peak_y") -> int | None:
    assert "selection" in x
    assert name in x["selection"]
    match len(x["selection"][name]):
        case 0:
            return None
        case 1:
            return x["selection"][name][0][value]


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


if __name__ == "__main__":
    app()
