import streamlit as st
import numpy as np
import pandas as pd
from src.cleaning import coerce_datetime, coerce_numeric, drop_duplicates, fill_missing, iqr_filter

st.title("02) Cleaning")
st.caption("Minimal, practical cleaning with a simple log.")

if "df" not in st.session_state:
    st.warning("No dataset. Go to 01 Data Upload.")
    st.stop()

df = st.session_state["df"]

st.subheader("1) Type coercion")
cols = df.columns.tolist()
date_col = st.selectbox("Datetime column (optional)", ["(none)"] + cols)
num_cols = st.multiselect("Numeric columns to coerce (optional)", cols)

colA, colB = st.columns(2)
with colA:
    numeric_strategy = st.selectbox("Fill missing for numeric", ["median", "mean", "zero"], index=0)
with colB:
    do_drop_dup = st.toggle("Drop duplicates", value=True)

st.subheader("2) Outlier removal (IQR)")
outlier_col = st.selectbox("Outlier filter column (optional)", ["(none)"] + cols)
iqr_k = st.slider("IQR k", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

if st.button("Apply cleaning", type="primary"):
    out = df.copy()
    log = st.session_state.setdefault("log", [])

    # coerce
    if date_col != "(none)":
        out = coerce_datetime(out, date_col)
        log.append(f"Coerced to datetime: {date_col}")

    for c in num_cols:
        out = coerce_numeric(out, c)
        log.append(f"Coerced to numeric: {c}")

    # fill missing
    before_missing = int(out.isna().sum().sum())
    out, rep = fill_missing(out, numeric_strategy=numeric_strategy)
    after_missing = int(out.isna().sum().sum())
    log.append(f"Filled missing (numeric={numeric_strategy}): numeric={rep['numeric_filled']}, categorical={rep['categorical_filled']} (remaining NA {after_missing})")

    # drop duplicates
    if do_drop_dup:
        out, removed = drop_duplicates(out)
        log.append(f"Dropped duplicates: {removed}")

    # iqr outliers
    if outlier_col != "(none)":
        out2, info = iqr_filter(out, outlier_col, k=iqr_k)
        out = out2
        log.append(f"IQR outlier filter on {outlier_col}: removed={info['removed']} (lower={info['lower']}, upper={info['upper']})")

    st.session_state["df"] = out
    st.success("Cleaning applied.")

st.markdown("---")
st.subheader("Current dataset")
st.write("Shape:", st.session_state["df"].shape)
st.dataframe(st.session_state["df"].head(30), use_container_width=True)

with st.expander("Log"):
    for item in st.session_state.get("log", []):
        st.write("•", item)
