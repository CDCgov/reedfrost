import streamlit as st

from reedfrost.app.charts import ui_charts
from reedfrost.app.controller import Controller
from reedfrost.app.inputs import (
    input_brn,
    input_metric,
    input_model,
    input_n,
    input_n_immune,
    input_n_infected,
    input_n_simulations,
    input_result_type,
    input_seed,
)


def ui(c: Controller):
    st.set_page_config(
        page_title="Chain binomial models", page_icon="🧮", layout="wide"
    )
    st.title("Chain binomial models")

    with st.sidebar:
        input_n(c)
        input_n_immune(c)
        input_brn(c)
        input_model(c)
        input_result_type(c)
        input_metric(c)

        st.header("Input parameters")
        with st.expander("Advanced options", expanded=False):
            input_n_infected(c)
            input_n_simulations(c)
            input_seed(c)

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    ui_charts(c)


if __name__ == "__main__":
    Controller(ui).run()
