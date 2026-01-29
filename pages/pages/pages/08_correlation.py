import streamlit as st
import numpy as np
from src.stats_tests import correlation
from src.viz import plot_scatter

st.title("08) Correlation")
st.caption("Correlation checks association (not causation).")

if "df" not in st.session_state:
    st.warning("No dataset. Go to 01 Data Upload.")
    st.stop()

df = st.session_state["df"]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.info("Need at least 2 numeric columns (try coercion in Cleaning).")
    st.stop()

method = st.selectbox("Method", ["pearson", "spearman"], index=0)
xcol = st.selectbox("X", numeric_cols, index=0)
ycol = st.selectbox("Y", numeric_cols, index=1)

alpha = st.slider("Significance level (alpha)", 0.001, 0.2, 0.05, 0.001)

if st.button("Run correlation", type="primary"):
    try:
        fig = plot_scatter(df[xcol], df[ycol], title=f"Scatter: {xcol} vs {ycol}")
        st.pyplot(fig, clear_figure=True)

        res = correlation(df[xcol], df[ycol], method=method)
        st.metric("r", f"{res['r']:.4f}")
        st.metric("p-value", f"{res['pvalue']:.6f}")

        if res["pvalue"] < alpha:
            st.success("Result: correlation is statistically significant.")
        else:
            st.info("Result: correlation not statistically significant.")

        st.write(res)
        st.warning("Even significant correlation does NOT prove causation.", icon="⚠️")
    except Exception as e:
        st.error(str(e))
