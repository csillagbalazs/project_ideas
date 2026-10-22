"""Estimators and output helpers for the record-earnings paper.

rd()      local-polynomial regression discontinuity with optional absorbed fixed effects (e.g.
          surprise bins), triangular or uniform kernel, two-way clustered standard errors.
diff_rd() difference in discontinuities at one cutoff between two groups of firms.
density() bunching test: Poisson regression of bin counts on a polynomial on each side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

IMPL = Path(__file__).resolve().parents[1]
OUT = IMPL / "output"
TAB, FIG, DATA = OUT / "tables", OUT / "figures", OUT / "data"
for p in (TAB, FIG, DATA):
    p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(IMPL.parents[1]))  # project_ideas, for the shared data and figure style
from common.plot import GRID, INK, INK2, SERIES, plt  # noqa: E402,F401

PCT = {"ar0", "car1_3", "car1_6", "car_pre", "mom"}  # outcomes reported in percent


def load() -> pd.DataFrame:
    return pd.read_parquet(DATA / "sample.parquet")


def savefig(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=120)
    plt.close(fig)


def macros(name: str, values: dict) -> None:
    """Write \\newcommand macros (letters-only names) so the paper never retypes a number."""
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    (TAB / f"macros_{name}.tex").write_text("\n".join(lines) + "\n")


def neg(s: str) -> str:
    return s.replace("-", "\\ensuremath{-}")


def fmt(x: float, d: int = 2) -> str:
    return neg(f"{x:,.{d}f}")


def stars(t: float) -> str:
    a = abs(t)
    return "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""


def cell(b: float, t: float, d: int = 2) -> tuple[str, str]:
    return neg(f"{b:.{d}f}") + stars(t), neg(f"({t:.2f})")


def write(name: str, text: str) -> None:
    (TAB / name).write_text(text)


def _groups(d: pd.DataFrame, cluster: tuple[str, ...]):
    if len(cluster) == 1:
        return pd.factorize(d[cluster[0]])[0]
    return np.column_stack([pd.factorize(d[c])[0] for c in cluster])


def _within(d: pd.DataFrame, cols: list[str], fe: str, w: pd.Series) -> pd.DataFrame:
    """Weighted within transformation: subtract weighted fixed-effect-group means (exact for one FE)."""
    z = d[cols].astype(float)
    sw = w.groupby(d[fe]).transform("sum")
    return z - z.mul(w, axis=0).groupby(d[fe]).transform("sum").div(sw, axis=0)


def rd(df: pd.DataFrame, y: str, run: str = "r", h: float = 1.0, fe: str | None = "sbin", poly: int = 1,
       kernel: str = "tri", donut: float = 0.0, cutoff: float = 0.0, extra: tuple[str, ...] = (),
       cluster: tuple[str, ...] = ("ann_mi", "permno")) -> dict:
    """Sharp RD at run = cutoff. Returns jump, kink (change in slope) and their t-statistics.

    With fe="sbin" every surprise bin has its own intercept, so the jump is identified only by firms
    with (almost) the same surprise that land on different sides of the cutoff.
    """
    need = [y, run, *extra, *cluster] + ([fe] if fe else [])
    d = df.loc[df[run].sub(cutoff).abs().le(h) & df[run].sub(cutoff).abs().gt(donut), list(dict.fromkeys(need))].dropna()
    x = d[run] - cutoff
    d = d.assign(D=(x > 0).astype(float), x=x, xD=x * (x > 0))
    cols = ["D", "x", "xD"]
    if poly == 2:
        d = d.assign(x2=x ** 2, x2D=x ** 2 * (x > 0))
        cols += ["x2", "x2D"]
    cols += list(extra)
    w = (1 - x.abs() / h) if kernel == "tri" else pd.Series(1.0, index=d.index)
    if fe:
        # Only groups with observations on both sides of the cutoff identify the jump.
        both = d.groupby(fe).D.transform(lambda s: 0 < s.mean() < 1)
        d, w = d[both], w[both]
        Z = _within(d, [y, *cols], fe, w)
        X = Z[cols]
    else:
        Z = d[[y, *cols]].astype(float)
        X = sm.add_constant(Z[cols])
    res = sm.WLS(Z[y], X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": _groups(d, cluster)})
    sc = 100.0 if y in PCT else 1.0
    below = d[(d.x < 0)]
    return {"jump": sc * res.params["D"], "se": sc * res.bse["D"], "t": res.tvalues["D"],
            "kink": sc * res.params["xD"], "kse": sc * res.bse["xD"], "kt": res.tvalues["xD"],
            "n": len(d), "mean_below": sc * np.average(below[y], weights=w[below.index]) if len(below) else np.nan,
            "res": res}


def diff_rd(df: pd.DataFrame, y: str, group: str, run: str = "sue", h: float = 1.0, fe: str | None = "ann_mi",
            extra: tuple[str, ...] = (), cluster: tuple[str, ...] = ("ann_mi", "permno")) -> dict:
    """Difference in discontinuities at run = 0 between group == 1 and group == 0.

    Every term of the local-linear RD (intercept, jump, slopes) is interacted with the group, so the
    group-difference coefficients (DG = jump, xDG = kink) compare the two groups at the same cutoff.
    """
    need = [y, run, group, *extra, *cluster] + ([fe] if fe else [])
    d = df.loc[df[run].abs().le(h), list(dict.fromkeys(need))].dropna()
    x, G = d[run], d[group].astype(float)
    D = (x > 0).astype(float)
    d = d.assign(D=D, x=x, xD=x * D, G=G, DG=D * G, xG=x * G, xDG=x * D * G)
    cols = ["D", "x", "xD", "G", "DG", "xG", "xDG"]
    for e in extra:
        d[e + "G"] = d[e] * G
        cols += [e, e + "G"]
    w = 1 - x.abs() / h
    if fe:
        Z = _within(d, [y, *cols], fe, w)
        X = Z[cols]
    else:
        Z = d[[y, *cols]].astype(float)
        X = sm.add_constant(Z[cols])
    res = sm.WLS(Z[y], X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": _groups(d, cluster)})
    sc = 100.0 if y in PCT else 1.0
    out = {"n": len(d), "nG": int(G.sum()), "res": res}
    for k in ["D", "xD", "DG", "xDG", "x", "xG", *extra]:
        out[k], out[k + "_t"] = sc * res.params[k], res.tvalues[k]
    return out


def density(x: np.ndarray, h: float = 1.0, bw: float = 0.05, poly: int = 1, cutoff: float = 0.0) -> dict:
    """Bunching test: Poisson regression of bin counts on a polynomial in the bin midpoint on each side.

    The jump exp(D) - 1 is the percentage excess of the density just above the cutoff.
    """
    x = np.asarray(x, float) - cutoff
    x = x[np.isfinite(x) & (x != 0)]
    edges = np.arange(-h, h + bw / 2, bw)
    cnt, _ = np.histogram(x, bins=edges)
    mid = (edges[:-1] + edges[1:]) / 2
    d = pd.DataFrame({"c": cnt, "x": mid, "D": (mid > 0).astype(float)})
    d["xD"] = d.x * d.D
    cols = ["D", "x", "xD"]
    if poly == 2:
        d["x2"], d["x2D"] = d.x ** 2, d.x ** 2 * d.D
        cols += ["x2", "x2D"]
    res = sm.GLM(d.c, sm.add_constant(d[cols]), family=sm.families.Poisson()).fit(cov_type="HC1")
    return {"jump": 100 * (np.exp(res.params.D) - 1), "t": res.tvalues.D, "n": int(cnt.sum()), "bins": d,
            "fit": res.predict(sm.add_constant(d[cols]))}


def binned(df: pd.DataFrame, y: str, run: str, width: float = 0.1, h: float = 2.0, fe: str | None = "sbin"):
    """Bin means of y (net of fixed effects, mean added back) by bins of the running variable."""
    d = df.loc[df[run].abs().le(h), [y, run] + ([fe] if fe else [])].dropna().copy()
    if fe:
        d[y] = d[y] - d.groupby(fe)[y].transform("mean") + d[y].mean()
    d["b"] = (np.floor(d[run] / width) + 0.5) * width
    g = d.groupby("b")[y]
    return g.mean(), g.sem(), d
