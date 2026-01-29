import streamlit as st
import numpy as np
import pandas as pd
from src.viz import plot_timeseries

st.title("07) Time Series")
st.caption("Trend + simple moving average + resampling.")

if "df" not in st.session_state:
    st.warning("No dataset. Go to 01 Data Upload.")
    st.stop()

df = st.session_state["df"]
cols = df.columns.tolist()

date_col = st.selectbox("Datetime column", cols)
value_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
value_col = st.selectbox("Value (numeric) column", value_candidates if value_candidates else cols)

ma = st.slider("Moving average window", 2, 60, 7)
freq = st.selectbox("Resample frequency", ["none", "D", "W", "M"], index=2)

if st.button("Plot", type="primary"):
    try:
        fig, ts = plot_timeseries(df, date_col, value_col, ma_window=ma, title=f"{value_col} over time")
        st.pyplot(fig, clear_figure=True)

        if freq != "none":
            resampled = ts[value_col].resample(freq).mean().to_frame("value")
            st.subheader(f"Resampled mean ({freq})")
            st.dataframe(resampled.tail(30), use_container_width=True)

        st.info("If you see repeating ups/downs at regular intervals, you likely have seasonality (周期性).")
    except Exception as e:
        st.error(str(e))
