import streamlit as st

from reedfrost.app.input import register_inputs
from reedfrost.app.model import get_results
from reedfrost.app.view import view


def run_app():
    inputter = register_inputs()

    # set up app input layout and collect input values ------------------------
    st.set_page_config(
        page_title="Chain binomial models", page_icon="🧮", layout="wide"
    )
    st.title("Chain binomial models")

    with st.sidebar:
        inputter.place_component("n")
        inputter.place_component("n_immune")
        inputter.place_component("brn")
        inputter.place_component("model")
        inputter.place_component("result_type")
        inputter.place_component("metric")

        st.header("Input parameters")
        with st.expander("Advanced options", expanded=False):
            inputter.place_component("n_infected")
            inputter.place_component("n_simulations")
            inputter.place_component("seed")

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    # run the simulations/computations ----------------------------------------
    results = get_results(inputter.inputs)

    # render the results ------------------------------------------------------
    view(inputter.inputs, results)


if __name__ == "__main__":
    run_app()
