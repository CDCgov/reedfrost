import altair as alt
import numpy as np
import polars as pl
import streamlit as st

import reedfrost as rf


def main_tab():
    n_susceptible = st.slider(
        "No. initially susceptible", min_value=0, max_value=50, step=1, value=10
    )
    n_infected = st.slider(
        "No. initially infected", min_value=1, max_value=10, step=1, value=1
    )
    p = (
        st.slider(
            "Prob. of infection",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%d%%",
            value=5.0,
        )
        / 100
    )

    k = np.array(range(n_susceptible + 1))
    dens = rf.pmf(k=k, n=n_susceptible, p=p, m=n_infected)

    st.altair_chart(
        alt.Chart(pl.DataFrame({"k": k, "dens": dens * 100}))
        .properties(title=f"R_eff={round(n_susceptible * p, 2)}")
        .encode(
            alt.X("k:N", title="Additional no. infected"),
            alt.Y("dens", title="Probability (%)"),
        )
        .mark_bar()
    )


def app():
    st.title("Reed-Frost model")
    tab1, tab2 = st.tabs(["App", "Model description"])

    with tab1:
        main_tab()

    with tab2:
        with open("docs/index.md") as f:
            docs = f.read()

        st.markdown(docs)


if __name__ == "__main__":
    app()
