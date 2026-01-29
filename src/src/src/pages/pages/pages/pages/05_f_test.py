import streamlit as st
import numpy as np
from src.stats_tests import f_test_variance

st.title("05) F-test (Variance comparison)")
st.caption("Check whether variability differs between two groups (normality assumed).")

if "df" not in st.session_state:
    st.warning("No dataset. Go to 01 Data Upload.")
    st.stop()

df = st.session_state["df"]

alpha = st.slider("Significance level (alpha)", 0.001, 0.2, 0.05, 0.001)

group_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.info("No numeric columns found (try coercion in Cleaning).")
    st.stop()

gcol = st.selectbox("Group column (A/B, Before/After)", group_cols)
vcol = st.selectbox("Numeric value column", numeric_cols)

groups = df[gcol].dropna().astype("string").unique().tolist()
groups = sorted(groups)

if len(groups) < 2:
    st.info("Need at least 2 groups in group column.")
    st.stop()

g1 = st.selectbox("Group 1", groups, index=0)
g2 = st.selectbox("Group 2", groups, index=1 if len(groups) > 1 else 0)

if st.button("Run F-test", type="primary"):
    try:
        x = df.loc[df[gcol].astype("string") == g1, vcol]
        y = df.loc[df[gcol].astype("string") == g2, vcol]
        res = f_test_variance(x, y)
        st.write(res)
        st.metric("p-value", f"{res['pvalue']:.6f}")
        if res["pvalue"] < alpha:
            st.success("Result: variances differ (significant).")
        else:
            st.info("Result: no evidence variances differ.")
    except Exception as e:
        st.error(str(e))
