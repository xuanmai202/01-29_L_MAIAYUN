from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def binomial_test(k: int, n: int, p0: float, alternative: str = "two-sided") -> dict:
    """
    alternative: 'two-sided', 'greater', 'less'
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0 <= k <= n):
        raise ValueError("k must be in [0, n]")
    if not (0 <= p0 <= 1):
        raise ValueError("p0 must be in [0, 1]")

    res = stats.binomtest(k=k, n=n, p=p0, alternative=alternative)
    return {
        "k": k,
        "n": n,
        "p0": p0,
        "rate": k / n,
        "pvalue": float(res.pvalue),
    }


def f_test_variance(x: pd.Series, y: pd.Series) -> dict:
    """
    Classical F-test for equality of variances (assumes normality).
    Returns two-sided p-value.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 observations per group.")

    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    if vx == 0 or vy == 0:
        raise ValueError("Variance is zero in one group; F-test not meaningful.")

    # F = larger variance / smaller variance for stability
    if vx >= vy:
        f = vx / vy
        dfn, dfd = len(x) - 1, len(y) - 1
    else:
        f = vy / vx
        dfn, dfd = len(y) - 1, len(x) - 1

    # two-sided p-value
    p_one_tail = 1 - stats.f.cdf(f, dfn, dfd)
    p_two = 2 * min(p_one_tail, stats.f.cdf(f, dfn, dfd))
    p_two = max(0.0, min(1.0, float(p_two)))

    return {
        "n1": int(len(x)),
        "n2": int(len(y)),
        "var1": float(vx),
        "var2": float(vy),
        "F": float(f),
        "dfn": int(dfn),
        "dfd": int(dfd),
        "pvalue": p_two,
    }


def t_test_independent(x: pd.Series, y: pd.Series, equal_var: bool = False) -> dict:
    """
    Independent two-sample t-test.
    equal_var=False uses Welch's t-test (recommended in practice).
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 observations per group.")

    t, p = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy="omit")
    return {
        "n1": int(len(x)),
        "n2": int(len(y)),
        "mean1": float(x.mean()),
        "mean2": float(y.mean()),
        "t": float(t),
        "pvalue": float(p),
        "equal_var": bool(equal_var),
    }


def correlation(x: pd.Series, y: pd.Series, method: str = "pearson") -> dict:
    """
    method: pearson or spearman
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        raise ValueError("Need at least 3 paired observations.")

    if method == "spearman":
        r, p = stats.spearmanr(df["x"], df["y"])
    else:
        r, p = stats.pearsonr(df["x"], df["y"])

    return {
        "n": int(len(df)),
        "r": float(r),
        "pvalue": float(p),
        "method": method,
    }


def chi_square_independence(df: pd.DataFrame, row_col: str, col_col: str) -> dict:
    """
    Chi-square test of independence using contingency table.
    """
    ct = pd.crosstab(df[row_col], df[col_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Contingency table must be at least 2x2.")

    chi2, p, dof, expected = stats.chi2_contingency(ct.values)
    expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)

    return {
        "table": ct,
        "expected": expected_df,
        "chi2": float(chi2),
        "pvalue": float(p),
        "dof": int(dof),
    }
