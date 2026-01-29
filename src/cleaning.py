from __future__ import annotations
import numpy as np
import pandas as pd


def coerce_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def coerce_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates()
    return out, before - len(out)


def fill_missing(df: pd.DataFrame, numeric_strategy: str = "median") -> tuple[pd.DataFrame, dict]:
    """
    numeric_strategy: 'median' or 'mean' or 'zero'
    categorical: fill with 'Unknown'
    """
    out = df.copy()
    report = {"numeric_filled": 0, "categorical_filled": 0}

    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in out.columns if c not in num_cols]

    # numeric
    for c in num_cols:
        missing = int(out[c].isna().sum())
        if missing == 0:
            continue
        if numeric_strategy == "mean":
            val = out[c].mean()
        elif numeric_strategy == "zero":
            val = 0
        else:
            val = out[c].median()
        out[c] = out[c].fillna(val)
        report["numeric_filled"] += missing

    # categorical
    for c in cat_cols:
        missing = int(out[c].isna().sum())
        if missing == 0:
            continue
        out[c] = out[c].fillna("Unknown")
        report["categorical_filled"] += missing

    return out, report


def iqr_filter(df: pd.DataFrame, col: str, k: float = 1.5) -> tuple[pd.DataFrame, dict]:
    """
    Remove outliers using IQR rule.
    Returns filtered df and stats.
    """
    out = df.copy()
    s = pd.to_numeric(out[col], errors="coerce")
    before = len(out)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return out, {"removed": 0, "lower": None, "upper": None}

    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (s >= lower) & (s <= upper)
    out = out.loc[mask].copy()
    removed = before - len(out)
    return out, {"removed": int(removed), "lower": float(lower), "upper": float(upper)}
