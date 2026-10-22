"""Econometric and output helpers for the bundled-reports paper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

IMPL = Path(__file__).resolve().parents[1]
OUT = IMPL / "output"
TAB = OUT / "tables"
FIG = OUT / "figures"
DATA = OUT / "data"
for p in (TAB, FIG, DATA):
    p.mkdir(parents=True, exist_ok=True)

import sys  # noqa: E402

sys.path.insert(0, str(IMPL.parents[1]))  # project_ideas, for the shared figure style
from common.plot import GRID, INK, INK2, SERIES, plt  # noqa: E402,F401

CLASS_ORDER = ["Separate months", "Same month", "Same day"]
CLASS_COLOR = {"Separate months": SERIES[2], "Same month": SERIES[1], "Same day": SERIES[0]}


def savefig(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=120)
    plt.close(fig)


def macros(name: str, values: dict) -> None:
    """Write \\newcommand macros (letters-only names) so the paper never retypes a number."""
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    (TAB / f"macros_{name}.tex").write_text("\n".join(lines) + "\n")


def fmt(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace("-", "\\ensuremath{-}") if x < 0 else f"{x:,.{d}f}"


def demean(df: pd.DataFrame, cols: list[str], fes: list[str], tol: float = 1e-8, max_iter: int = 200) -> pd.DataFrame:
    """Absorb one or more sets of fixed effects by alternating projections."""
    out = df[cols].astype(float).copy()
    if not fes:
        return out
    groups = [df[f].to_numpy() for f in fes]
    for _ in range(max_iter):
        change = 0.0
        for g in groups:
            means = out.groupby(g).transform("mean")
            out = out - means
            change = max(change, float(np.nanmax(np.abs(means.to_numpy()))))
        if len(fes) == 1 or change < tol:
            break
    return out


def partial_group_slopes(df: pd.DataFrame, cols: list[str], blocks: list[tuple[str, list[str]]],
                         iters: int = 60, tol: float = 1e-9) -> pd.DataFrame:
    """Residualise `cols` on group-specific intercepts AND slopes.

    Each block (group, regs) lets every group have its own intercept and its own coefficients
    on `regs`, e.g. ("d4", ["s4", "s1"]) gives each annual-report date its own response to both
    surprises. Several blocks are combined by alternating projections (exact at convergence).
    """
    out = df[cols].astype(float).to_numpy().copy()
    projs = []
    for g, regs in blocks:
        X = np.column_stack([np.ones(len(df)), df[regs].astype(float).to_numpy()]) if regs else np.ones((len(df), 1))
        idx = pd.Series(np.arange(len(df))).groupby(df[g].to_numpy()).indices
        projs.append([(i, X[i] @ np.linalg.pinv(X[i])) for i in idx.values()])
    for _ in range(iters if len(blocks) > 1 else 1):
        before = out.copy()
        for blk in projs:
            for i, P in blk:
                out[i] = out[i] - P @ out[i]
        if len(blocks) == 1 or np.nanmax(np.abs(out - before)) < tol:
            break
    return pd.DataFrame(out, index=df.index, columns=cols)


def reg_slopes(df: pd.DataFrame, y: str, xs: list[str], blocks: list[tuple[str, list[str]]],
               cluster: list[str]):
    """OLS of y on xs after partialling group-specific intercepts and slopes (see above)."""
    need = list(dict.fromkeys([y, *xs, *[b[0] for b in blocks], *[r for b in blocks for r in b[1]], *cluster]))
    d = df[need].dropna().copy()
    z = partial_group_slopes(d, [y, *xs], blocks)
    keep = (z[xs].abs().sum(axis=1) > 1e-10)  # rows with no remaining variation carry no information
    z, d = z[keep], d[keep]
    if len(cluster) == 1:
        g = pd.factorize(d[cluster[0]])[0]
    else:
        g = np.column_stack([pd.factorize(d[c])[0] for c in cluster])
    res = sm.OLS(z[y], z[xs]).fit(cov_type="cluster", cov_kwds={"groups": g})
    res.nobs_used = len(d)
    return res


def reg(df: pd.DataFrame, y: str, xs: list[str], fes: list[str] | None = None,
        cluster: list[str] | None = None) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS with absorbed fixed effects and one- or two-way clustered standard errors."""
    fes = fes or []
    cluster = cluster or []
    need = list(dict.fromkeys([y, *xs, *fes, *cluster]))
    d = df[need].dropna().copy()
    # Drop singletons of each FE (they carry no within variation).
    for f in fes:
        d = d[d.groupby(f)[f].transform("size") > 1]
    z = demean(d, [y, *xs], fes)
    X = z[xs] if fes else sm.add_constant(z[xs])
    if len(cluster) == 0:
        res = sm.OLS(z[y], X).fit(cov_type="HC1")
    elif len(cluster) == 1:
        res = sm.OLS(z[y], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster[0]])[0]})
    else:
        g = np.column_stack([pd.factorize(d[c])[0] for c in cluster])
        res = sm.OLS(z[y], X).fit(cov_type="cluster", cov_kwds={"groups": g})
    res.nobs_used = len(d)
    return res


def stars(t: float) -> str:
    a = abs(t)
    return "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""


def coef_cell(res, var: str, scale: float = 1.0, digits: int = 2) -> tuple[str, str]:
    if var not in res.params.index:
        return "", ""
    b, t = res.params[var] * scale, res.tvalues[var]
    neg = lambda x: x.replace("-", "\\ensuremath{-}")
    return neg(f"{b:.{digits}f}") + stars(t), neg(f"({t:.2f})")


def regtable(models: list, rows: list[tuple[str, str]], scale: float = 1.0, digits: int = 2,
             extra: list[tuple[str, list[str]]] | None = None, header: list[str] | None = None) -> str:
    """Booktabs body: coefficient rows with t-stats below, then extra rows (FE, N, ...)."""
    lines = []
    if header:
        lines.append(" & " + " & ".join(header) + r"\\")
        lines.append(r"\midrule")
    for var, label in rows:
        cells = [coef_cell(m, var, scale, digits) for m in models]
        if all(c[0] == "" for c in cells):
            continue
        lines.append(label + " & " + " & ".join(c[0] for c in cells) + r"\\")
        lines.append(" & " + " & ".join(c[1] for c in cells) + r"\\[1pt]")
    lines.append(r"\midrule")
    for label, vals in (extra or []):
        lines.append(label + " & " + " & ".join(vals) + r"\\")
    lines.append("Observations & " + " & ".join(f"{m.nobs_used:,}" for m in models) + r"\\")
    return "\n".join(lines)


def write(name: str, text: str) -> None:
    (TAB / name).write_text(text)


def wald_diff(res, a: str, b: str) -> tuple[float, float]:
    """Difference of two coefficients and its t-statistic."""
    r = np.zeros(len(res.params))
    idx = list(res.params.index)
    r[idx.index(a)] = 1
    r[idx.index(b)] = -1
    t = res.t_test(r)
    return float(np.squeeze(t.effect)), float(np.squeeze(t.tvalue))
