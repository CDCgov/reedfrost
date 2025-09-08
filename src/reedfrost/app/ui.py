import streamlit as st

from reedfrost.app.charts import ui_charts
from reedfrost.app.controller import Controller


def app(c: Controller):
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


components = [
    {
        "type": "input",
        "key": "n",
        "setter": lambda c: st.slider(
            "Population size", min_value=1, max_value=100, step=1, value=10
        ),
    },
    {
        "type": "input",
        "key": "n_immune",
        "setter": lambda c: st.select_slider(
            "Proportion initially immune",
            # values are from 0 to N-1, leaving space for at least 1 infected
            options=range(0, c.get("n")),
            value=0,
            format_func=lambda x: f"{x / c.get('n'):.0%}",
        ),
    },
    {
        "type": "input",
        "key": "brn",
        "setter": lambda c: st.slider(
            "Basic reproduction number",
            min_value=0.0,
            max_value=min(15.0, float(c.get("n"))),
            step=0.1,
            value=min(1.5, float(c.get("n"))),
            format="%.1f",
        ),
    },
    {
        "type": "input",
        "key": "model",
        "setter": lambda c: st.selectbox(
            "Model",
            options=["Reed-Frost", "Enko", "Greenwood"],
            index=0,
        ),
    },
    {
        "type": "input",
        "key": "result_type",
        "setter": lambda c: st.selectbox(
            "Results type", options=["Trajectories", "Theoretical"], index=0
        ),
    },
    {
        "type": "input",
        "key": "metric",
        "setter": lambda c: st.selectbox(
            "Infections metric", options=["Cumulative", "Incident"], index=0
        ),
    },
    {
        "type": "input",
        "key": "n_simulations",
        "setter": lambda c: st.slider(
            "No. simulations", min_value=5, max_value=250, step=1, value=100
        ),
    },
    {
        "type": "input",
        "key": "seed",
        "setter": lambda c: st.number_input(
            "Random seed", min_value=0, max_value=2**32 - 1, step=1, value=42
        ),
    },
    {"type": "input", "key": "n_infected", "setter": set_n_infected},
    {"key": "charts", "func": ui_charts},
]
