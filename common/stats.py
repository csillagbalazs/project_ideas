"""Small econometrics helpers: Fama-MacBeth, clustered OLS, portfolio alphas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def nw_mean(x: pd.Series, lags: int = 3) -> tuple[float, float]:
    """Mean and Newey-West t-statistic of a time series."""
    x = pd.Series(x).dropna()
    r = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.tvalues[0])


def fama_macbeth(df: pd.DataFrame, y: str, xs: list[str], t: str = "mi", lags: int = 3,
                 min_obs: int = 30) -> pd.DataFrame:
    """Cross-sectional OLS per period; NW t-stats on the coefficient series."""
    rows = []
    for per, g in df[[t, y, *xs]].dropna().groupby(t):
        if len(g) < max(min_obs, len(xs) + 5):
            continue
        X = sm.add_constant(g[xs].to_numpy(), has_constant="add")
        b = np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0]
        rows.append([per, len(g), *b])
    c = pd.DataFrame(rows, columns=[t, "n", "const", *xs])
    out = []
    for v in ["const", *xs]:
        m, tt = nw_mean(c[v], lags)
        out.append({"var": v, "coef": m, "t": tt})
    res = pd.DataFrame(out).set_index("var")
    res.attrs["periods"] = len(c)
    res.attrs["avg_n"] = float(c.n.mean())
    res.attrs["series"] = c
    return res


def ols_cluster(df: pd.DataFrame, y: str, xs: list[str], cluster: str, fe: str | None = None):
    """OLS with one-way clustered SEs; optional fixed effects by demeaning."""
    d = df[[y, *xs, cluster] + ([fe] if fe and fe != cluster else [])].dropna().copy()
    if fe:
        for v in [y, *xs]:
            d[v] = d[v] - d.groupby(fe)[v].transform("mean")
        X = d[xs]
    else:
        X = sm.add_constant(d[xs])
    return sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})


def alpha(ret: pd.Series, fac: pd.DataFrame, model: str = "CH4", lags: int = 3):
    """Time-series alpha of a monthly (excess or long-short) return series on LSY factors."""
    cols = {"CAPM": ["mktrf"], "CH3": ["mktrf", "SMB", "VMG"], "CH4": ["mktrf", "SMB", "VMG", "PMO"]}[model]
    d = pd.concat([ret.rename("y"), fac.set_index("mi")[cols]], axis=1, join="inner").dropna()
    r = sm.OLS(d.y, sm.add_constant(d[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params["const"]), float(r.tvalues["const"]), len(d)


def sort_portfolios(df: pd.DataFrame, signal: str, ret: str, n: int = 5, t: str = "mi",
                    weight: str | None = None) -> pd.DataFrame:
    """Monthly quantile portfolios on `signal`; returns a period x quantile return table."""
    d = df[[t, signal, ret] + ([weight] if weight else [])].dropna().copy()
    d["q"] = d.groupby(t)[signal].transform(
        lambda s: pd.qcut(s.rank(method="first"), n, labels=False) + 1 if len(s) >= 5 * n else np.nan)
    d = d.dropna(subset=["q"])
    if weight:
        f = lambda g: np.average(g[ret], weights=g[weight])
        p = d.groupby([t, "q"]).apply(f, include_groups=False).unstack()
    else:
        p = d.groupby([t, "q"])[ret].mean().unstack()
    p["HL"] = p[n] - p[1]
    return p


def fmt_table(res: pd.DataFrame, names: dict | None = None, digits: int = 3) -> str:
    """Coefficient (t-stat) lines for quick LaTeX tables."""
    names = names or {}
    lines = []
    for v, r in res.iterrows():
        lines.append(f"{names.get(v, v)} & {r.coef:.{digits}f} & ({r.t:.2f})\\\\")
    return "\n".join(lines)
