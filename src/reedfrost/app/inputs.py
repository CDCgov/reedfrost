import streamlit as st

from reedfrost.app.controller import Controller


def input_n(c: Controller) -> None:
    c.set(
        "n", st.slider("Population size", min_value=1, max_value=100, step=1, value=10)
    )


def input_n_immune(c: Controller) -> None:
    n = c.get("n")
    assert isinstance(n, int) and n >= 1
    c.set(
        "n_immune",
        st.select_slider(
            "Proportion initially immune",
            # values are from 0 to N-1, leaving space for at least 1 infected
            options=range(0, n),
            value=0,
            format_func=lambda x: f"{x / c.get('n'):.0%}",
        ),
    )


def input_brn(c: Controller) -> None:
    n = c.get("n")
    assert isinstance(n, int) and n >= 1
    c.set(
        "brn",
        st.slider(
            "Basic reproduction number",
            min_value=0.0,
            max_value=min(15.0, float(n)),
            step=0.1,
            value=min(1.5, float(n)),
            format="%.1f",
        ),
    )


def input_model(c: Controller) -> None:
    c.set(
        "model",
        st.selectbox("Model", options=["Reed-Frost", "Enko", "Greenwood"], index=0),
    )


def input_result_type(c: Controller) -> None:
    c.set(
        "result_type",
        st.selectbox("Results type", options=["Trajectories", "Theoretical"], index=0),
    )


def input_metric(c: Controller) -> None:
    c.set(
        "metric",
        st.selectbox("Infections metric", options=["Cumulative", "Incident"], index=0),
    )


def input_n_simulations(c: Controller) -> None:
    c.set(
        "n_simulations",
        st.slider("No. simulations", min_value=5, max_value=250, step=1, value=100),
    )


def input_seed(c: Controller) -> None:
    c.set(
        "seed",
        st.number_input(
            "Random seed", min_value=0, max_value=2**32 - 1, step=1, value=42
        ),
    )


def input_n_infected(c: Controller) -> None:
    # need special handling for the case where everyone is immune but 1,
    # because streamlit sliders must have a range
    n = c.get("n")
    n_immune = c.get("n_immune")
    assert isinstance(n, int) and n >= 1
    assert isinstance(n_immune, int) and 0 <= n_immune < n

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

    c.set("n_infected", n_infected)
