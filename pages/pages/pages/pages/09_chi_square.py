import streamlit as st
import pandas as pd
from src.stats_tests import chi_square_independence

st.title("09) Chi-square Test (Independence)")
st.caption("Test whether two categorical variables are independent.")

if "df" not in st.session_state:
    st.warning("No dataset. Go to 01 Data Upload.")
    st.stop()

df = st.session_state["df"]
cols = df.columns.tolist()

alpha = st.slider("Significance level (alpha)", 0.001, 0.2, 0.05, 0.001)

row_col = st.selectbox("Row category column", cols, index=0)
col_col = st.selectbox("Column category column", cols, index=1 if len(cols) > 1 else 0)

if st.button("Run chi-square", type="primary"):
    try:
        res = chi_square_independence(df, row_col, col_col)

        st.subheader("Observed (contingency table)")
        st.dataframe(res["table"], use_container_width=True)

        st.subheader("Expected counts")
        st.dataframe(res["expected"].round(3), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("chi2", f"{res['chi2']:.4f}")
        c2.metric("dof", f"{res['dof']}")
        c3.metric("p-value", f"{res['pvalue']:.6f}")

        if res["pvalue"] < alpha:
            st.success("Result: evidence of association (not independent).")
        else:
            st.info("Result: no evidence of association (independence plausible).")
    except Exception as e:
        st.error(str(e))
