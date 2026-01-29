import streamlit as st
import numpy as np
from src.stats_tests import t_test_independent

st.title("06) T-test (Effect: mean difference)")
st.caption("Independent two-sample t-test (Welch recommended).")

if "df" not in st.session_state:
    st.warning("No dataset. Go to 01 Data Upload.")
    st.stop()

df = st.session_state["df"]

alpha = st.slider("Significance level (alpha)", 0.001, 0.2, 0.05, 0.001)
equal_var = st.toggle("Assume equal variances (classic t-test)", value=False)

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

if st.button("Run T-test", type="primary"):
    try:
        x = df.loc[df[gcol].astype("string") == g1, vcol]
        y = df.loc[df[gcol].astype("string") == g2, vcol]
        res = t_test_independent(x, y, equal_var=equal_var)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"mean({g1})", f"{res['mean1']:.4f}")
        c2.metric(f"mean({g2})", f"{res['mean2']:.4f}")
        c3.metric("p-value", f"{res['pvalue']:.6f}")

        if res["pvalue"] < alpha:
            st.success("Result: mean difference is statistically significant.")
        else:
            st.info("Result: no evidence of mean difference.")

        st.write(res)
    except Exception as e:
        st.error(str(e))
