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


# Special purpose component functions ---------------------------------------------------
def set_n_infected(c: Controller) -> int:
    # need special handling for the case where everyone is immune but 1,
    # because streamlit sliders must have a range
    n = c.get("n")
    n_immune = c.get("n_immune")
    assert isinstance(n, int) and n >= 1
    assert isinstance(n_immune, int) and 0 <= n_immune < n

    if n - n_immune == 1:
        st.text("No. initially infected: 1")
        return 1
    else:
        return st.slider(
            "No. initially infected",
            min_value=1,
            max_value=n - n_immune,
            step=1,
            value=1,
        )


def selectbox(label, options, value, **kwargs):
    return st.selectbox(label, options=options, index=options.index(value), **kwargs)


# UI components ------------------------------------------------------------------------

# Components each have:
# - `key`: unique identifier
# - `type`: "input" or "output"
# - if `type` is "input", a `setter` function that takes a Controller and returns a
#   value to set for that key
# - if `type` is "output", a `func` function that takes a Controller and produces output
COMPONENTS = [
    {
        "key": "n",
        "type": "input",
        "setter": lambda c: st.slider(
            "Population size", min_value=1, max_value=100, step=1, value=c.get("n")
        ),
    },
    {
        "key": "n_immune",
        "type": "input",
        "setter": lambda c: st.select_slider(
            "Proportion initially immune",
            # values are from 0 to N-1, leaving space for at least 1 infected
            options=range(0, c.get("n")),
            value=c.get("n_immune"),
            format_func=lambda x: f"{x / c.get('n'):.0%}",
        ),
    },
    {
        "key": "brn",
        "type": "input",
        "setter": lambda c: st.slider(
            "Basic reproduction number",
            min_value=0.0,
            max_value=min(15.0, float(c.get("n"))),
            step=0.1,
            value=min(c.get("brn"), float(c.get("n"))),
            format="%.1f",
        ),
    },
    {
        "key": "model",
        "type": "input",
        "setter": lambda c: selectbox(
            "Model", options=["Reed-Frost", "Enko", "Greenwood"], value=c.get("model")
        ),
    },
    {
        "key": "result_type",
        "type": "input",
        "setter": lambda c: selectbox(
            "Results type",
            options=["Trajectories", "Theoretical"],
            value=c.get("result_type"),
        ),
    },
    {
        "key": "metric",
        "type": "input",
        "setter": lambda c: selectbox(
            "Infections metric",
            options=["Cumulative", "Incident"],
            value=c.get("metric"),
        ),
    },
    {
        "key": "n_simulations",
        "type": "input",
        "setter": lambda c: st.slider(
            "No. simulations",
            min_value=5,
            max_value=250,
            step=1,
            value=c.get("n_simulations"),
        ),
    },
    {
        "key": "seed",
        "type": "input",
        "setter": lambda c: st.number_input(
            "Random seed", min_value=0, max_value=2**32 - 1, step=1, value=c.get("seed")
        ),
    },
    {"type": "input", "key": "n_infected", "setter": set_n_infected},
    {"key": "charts", "type": "output", "func": ui_charts},
]
