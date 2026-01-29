from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram(series: pd.Series, bins: int = 20, title: str = "Histogram"):
    s = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots()
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def pareto_table(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    counts = df[category_col].astype("string").fillna("Unknown").value_counts(dropna=False)
    out = counts.rename("count").to_frame()
    out["ratio"] = out["count"] / out["count"].sum()
    out["cum_ratio"] = out["ratio"].cumsum()
    out.index.name = category_col
    return out.reset_index()


def plot_pareto(pareto_df: pd.DataFrame, category_col: str, title: str = "Pareto"):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    x = np.arange(len(pareto_df))
    ax1.bar(x, pareto_df["count"])
    ax2.plot(x, pareto_df["cum_ratio"])

    ax1.set_title(title)
    ax1.set_xlabel(category_col)
    ax1.set_ylabel("count")
    ax2.set_ylabel("cumulative ratio")

    ax1.set_xticks(x)
    ax1.set_xticklabels(pareto_df[category_col].astype(str), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_scatter(x: pd.Series, y: pd.Series, title: str = "Scatter"):
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                       "y": pd.to_numeric(y, errors="coerce")}).dropna()
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    return fig


def plot_timeseries(df: pd.DataFrame, date_col: str, value_col: str, ma_window: int = 7, title: str = "Time Series"):
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    tmp = tmp.set_index(date_col)

    fig, ax = plt.subplots()
    ax.plot(tmp.index, tmp[value_col], label="value")
    if ma_window and ma_window >= 2:
        ax.plot(tmp.index, tmp[value_col].rolling(ma_window).mean(), label=f"MA({ma_window})")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel(value_col)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, tmp
