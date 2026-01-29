import streamlit as st
import pandas as pd
from src.io import read_csv_bytes

st.title("01) Data Upload")
st.caption("Upload CSV and keep it in session_state.")

uploaded = st.file_uploader("CSV file", type=["csv"])
use_sample = st.toggle("Use sample.csv (data/sample.csv)", value=False)

if use_sample:
    try:
        df = pd.read_csv("data/sample.csv")
        st.session_state["df"] = df
        st.session_state.setdefault("log", []).append("Loaded sample.csv")
        st.success("Loaded sample.csv into session.")
    except Exception as e:
        st.error(f"Failed to load sample.csv: {e}")

if uploaded:
    try:
        raw = uploaded.getvalue()
        df = read_csv_bytes(raw)
        st.session_state["df"] = df
        st.session_state.setdefault("log", []).append(f"Uploaded CSV: {uploaded.name}")
        st.success("Uploaded data loaded into session.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]
    st.write("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Shape:", df.shape)

    with st.expander("Column dtypes"):
        st.dataframe(df.dtypes.astype(str).to_frame("dtype"), use_container_width=True)
else:
    st.info("Upload a CSV or toggle sample.")
