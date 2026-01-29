import streamlit as st

st.set_page_config(
    page_title="Stats Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Stats Dashboard")
st.caption("Practical template: CSV → Cleaning → Visualize → Tests")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset loaded", "Yes" if "df" in st.session_state else "No")
with col2:
    st.metric("Rows", int(st.session_state["df"].shape[0]) if "df" in st.session_state else 0)
with col3:
    st.metric("Columns", int(st.session_state["df"].shape[1]) if "df" in st.session_state else 0)

st.markdown("---")

st.subheader("How to use")
st.markdown(
    """
1. Go to **01 Data Upload** and upload your CSV  
2. Go to **02 Cleaning** to fix missing values / types / duplicates / outliers  
3. Use **03 Visualize** for histogram & pareto  
4. Run tests from **04–09** pages
"""
)

st.info("Tip: If you don’t have data, use `data/sample.csv` (included).", icon="💡")

if "log" in st.session_state and st.session_state["log"]:
    with st.expander("📌 Cleaning / Processing Log"):
        for item in st.session_state["log"]:
            st.write("•", item)
