import streamlit as st

from reedfrost.app.charts import ui_charts
from reedfrost.app.controller import Controller


def app(c: Controller):
    """Run a streamlit app"""

    st.set_page_config(
        page_title="Chain binomial models", page_icon="🧮", layout="wide"
    )
    st.title("Chain binomial models")

    with st.sidebar:
        c.place("n")
        c.place("n_immune")
        c.place("brn")
        c.place("model")
        c.place("result_type")
        c.place("metric")

        st.header("Input parameters")
        with st.expander("Advanced options", expanded=False):
            c.place("n_infected")
            c.place("n_simulations")
            c.place("seed")

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    c.place("charts")


# Special purpose UI component functions -------------------------------------------------
def n_immune(c: Controller, key: str, label: str, **kwargs):
    # ensure that n_immune does not change when `options` changes
    # see <https://docs.streamlit.io/develop/concepts/architecture/widget-behavior#retain-statefulness-when-changing-a-widgets-parameters>
    c.ensure(key)

    n = c.get("n")
    assert isinstance(n, int) and n >= 1

    st.select_slider(
        label,
        key=key,
        # values are from 0 to N-1, leaving space for at least 1 infected
        options=range(0, n),
        format_func=lambda x: f"{x / n:.0%}",
        **kwargs,
    )


def n_infected(c: Controller, key: str, label: str, **kwargs):
    # need special handling for the case where everyone is immune but 1,
    # because streamlit sliders must have a range
    n = c.get("n")
    n_immune = c.get("n_immune")

    assert isinstance(n, int) and n >= 1
    assert isinstance(n_immune, int) and 0 <= n_immune < n

    max_value = n - n_immune

    help = f"No. infected is at most total no. ({n}) minus no. immune ({n_immune}) = {max_value}"

    if max_value == 1:
        st.text(f"{label}: 1", help=help)
        c.set(key, 1)
    else:
        st.slider(label, key=key, max_value=max_value, help=help, **kwargs)


def brn(c: Controller, key: str, label: str, max_value: float, **kwargs):
    # ensure that R0 does not exceed N
    n = float(c.get("n"))
    c.set(key, min(c.get(key), n))
    st.slider(label, key=key, max_value=min(max_value, n), **kwargs)


# UI components ------------------------------------------------------------------------

COMPONENTS = [
    {
        "key": "n",
        "type": "input",
        "func": st.slider,
        "label": "Population size",
        "min_value": 3,
        "max_value": 100,
        "step": 1,
    },
    {
        "key": "n_immune",
        "type": "special_input",
        "label": "Proportion initially immune",
        "func": n_immune,
    },
    {
        "key": "brn",
        "type": "special_input",
        "func": brn,
        "label": "Basic reproduction number",
        "min_value": 0.0,
        "max_value": 15.0,
        "step": 0.1,
        "format": "%.1f",
    },
    {
        "key": "model",
        "type": "input",
        "func": st.radio,
        "label": "Model",
        "options": ["Reed-Frost", "Enko", "Greenwood"],
    },
    {
        "key": "result_type",
        "type": "input",
        "func": st.radio,
        "label": "Results type",
        "options": ["Trajectories", "Theoretical"],
    },
    {
        "key": "metric",
        "type": "input",
        "func": st.radio,
        "label": "Infections metric",
        "options": ["Cumulative", "Incident"],
    },
    {
        "key": "n_simulations",
        "type": "input",
        "func": st.slider,
        "label": "No. simulations",
        "min_value": 5,
        "max_value": 250,
        "step": 1,
    },
    {
        "key": "seed",
        "type": "input",
        "func": st.number_input,
        "label": "Random seed",
        "min_value": 0,
        "max_value": 2**32 - 1,
        "step": 1,
    },
    {
        "key": "n_infected",
        "type": "special_input",
        "func": n_infected,
        "label": "No. initially infected",
        "min_value": 1,
        "step": 1,
    },
    {"key": "charts", "type": "output", "func": ui_charts},
]
