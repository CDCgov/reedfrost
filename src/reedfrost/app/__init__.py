import streamlit as st

from reedfrost.app.input import register_inputs
from reedfrost.app.model import get_results
from reedfrost.app.view import view


def run_app():
    app = App()

    # set up app input layout and collect input values ------------------------
    st.set_page_config(
        page_title="Chain binomial models", page_icon="🧮", layout="wide"
    )
    st.title("Chain binomial models")

    with st.sidebar:
        app.place_component("n")
        app.place_component("n_immune")
        app.place_component("brn")
        app.place_component("model")
        app.place_component("result_type")
        app.place_component("metric")

        st.header("Input parameters")
        with st.expander("Advanced options", expanded=False):
            app.place_component("n_infected")
            app.place_component("n_simulations")
            app.place_component("seed")

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    # run the simulations/computations ----------------------------------------
    app.state["results"] = get_results(app.state)

    # render the results ------------------------------------------------------
    view(app.state)


if __name__ == "__main__":
    run_app()
