import altair as alt
import numpy as np
import polars as pl
import streamlit as st

import reedfrost as rf


def app():
    st.title("Reed-Frost model")

    with st.sidebar:
        st.header("Input parameters")
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

        st.text(f"R_eff = {round(n_susceptible * p, 2)}")

        st.divider()
        st.header("Links")

        st.page_link("https://github.com/CDCgov/reedfrost/", label="repo", icon="🗂️")
        st.page_link(
            "https://cdcgov.github.io/reedfrost/", label="documentation", icon="📝"
        )

    k = np.array(range(n_susceptible + 1))
    dens = rf.pmf(k=k, s=n_susceptible, i=n_infected, p=p)

    st.altair_chart(
        alt.Chart(pl.DataFrame({"k": k, "dens": dens * 100}))
        .properties(title="Final size distribution")
        .configure_title(anchor="middle")
        .encode(
            alt.X("k:N", title="Additional no. infected"),
            alt.Y("dens", title="Probability (%)"),
        )
        .mark_bar()
    )


if __name__ == "__main__":
    app()
