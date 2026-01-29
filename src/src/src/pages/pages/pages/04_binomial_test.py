import streamlit as st
import numpy as np
from src.stats_tests import binomial_test

st.title("04) Binomial Test (2-choice success rate)")
st.caption("Test whether success rate differs from p0.")

alpha = st.slider("Significance level (alpha)", 0.001, 0.2, 0.05, 0.001)
alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"], index=0)

col1, col2, col3 = st.columns(3)
with col1:
    k = st.number_input("Success count k", min_value=0, value=50, step=1)
with col2:
    n = st.number_input("Trials n", min_value=1, value=100, step=1)
with col3:
    p0 = st.number_input("Baseline p0", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Run binomial test", type="primary"):
    try:
        res = binomial_test(int(k), int(n), float(p0), alternative=alternative)
        st.metric("Observed rate", f"{res['rate']:.4f}")
        st.metric("p-value", f"{res['pvalue']:.6f}")
        if res["pvalue"] < alpha:
            st.success("Result: statistically significant (reject H0).")
        else:
            st.info("Result: not statistically significant (fail to reject H0).")
    except Exception as e:
        st.error(str(e))
