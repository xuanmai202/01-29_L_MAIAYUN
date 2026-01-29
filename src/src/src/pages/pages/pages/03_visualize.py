import streamlit as st
import numpy as np
from src.viz import plot_histogram, pareto_table, plot_pareto

st.title("03) Visualize")
st.caption("Histogram (numeric) and Pareto (category).")

if "df" not in st.session_state:
    st.warning("No dataset. Go to 01 Data Upload.")
    st.stop()

df = st.session_state["df"]
cols = df.columns.tolist()

tab1, tab2 = st.tabs(["Histogram", "Pareto"])

with tab1:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found (try coercion in Cleaning).")
    else:
        col = st.selectbox("Numeric column", numeric_cols)
        bins = st.slider("Bins", 5, 80, 20)
        fig = plot_histogram(df[col], bins=bins, title=f"Histogram: {col}")
        st.pyplot(fig, clear_figure=True)

        s = df[col].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("count", int(s.shape[0]))
        c2.metric("mean", float(s.mean()))
        c3.metric("median", float(s.median()))

with tab2:
    cat_col = st.selectbox("Category column", cols)
    p_df = pareto_table(df, cat_col)

    top_n = st.slider("Show top N categories", 5, min(50, len(p_df)), min(20, len(p_df)))
    show_df = p_df.head(top_n).copy()

    fig = plot_pareto(show_df, cat_col, title=f"Pareto: {cat_col} (top {top_n})")
    st.pyplot(fig, clear_figure=True)
    st.dataframe(show_df, use_container_width=True)

    # quick 80/20 insight
    hit = show_df[show_df["cum_ratio"] <= 0.8].shape[0]
    st.info(f"Top ~{hit} categories cover ~80% (within displayed top N).")
