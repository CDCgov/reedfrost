import streamlit as st

from reedfrost.app.charts import ui_charts
from reedfrost.app.controller import Controller
from reedfrost.app.inputs import inputs


def ui(c: Controller):
    st.set_page_config(
        page_title="Chain binomial models", page_icon="🧮", layout="wide"
    )
    st.title("Chain binomial models")

    with st.sidebar:
        c.place_input("n")
        c.place_input("n_immune")
        c.place_input("brn")
        c.place_input("model")
        c.place_input("result_type")
        c.place_input("metric")

        st.header("Input parameters")
        with st.expander("Advanced options", expanded=False):
            c.place_input("n_infected")
            c.place_input("n_simulations")
            c.place_input("seed")

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    ui_charts(c)


if __name__ == "__main__":
    Controller(ui=ui, inputs=inputs).run()
