"""Idea 3 PoC: are record earnings a reference point?

Running variable: r = (TTM EPS after this report - highest TTM EPS before it) / SD of the firm's
past SRW errors, where TTM EPS is the sum of the last four standalone quarters (bonus-adjusted, so
comparable within firm) and the record needs at least eight earlier TTM values. r > 0 means the
report sets a new all-time trailing-earnings record.

Regression discontinuity (local linear, triangular kernel, SEs clustered by month): is there a jump
at r = 0 in the announcement-month return or in the three-month drift? Validity checks:
covariate balance (surprise, size, momentum, pre-announcement return) and a density test for
bunching just above the record (earnings management, as at the thresholds of Degeorge et al., 1999).
Benchmark: the same design at the classic threshold SUE = 0 ("beat last year's quarter").
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

a = announcements().sort_values(["permno", "fyear", "fqtr"]).copy()
a["t"] = a.fyear * 4 + a.fqtr
g = a.groupby("permno", sort=False)
a["ttm"] = g.eps_actual.transform(lambda s: s.rolling(4, min_periods=4).sum())
a.loc[(a.t - g.t.shift(3)) != 3, "ttm"] = np.nan          # last four quarters must be consecutive
a["prevmax"] = g.ttm.transform(lambda s: s.shift(1).expanding(min_periods=8).max())
a["r"] = (a.ttm - a.prevmax) / a.sd_ue
a = a[a.year.between(2005, 2020) & np.isfinite(a.r) & (a.prevmax > 0)].copy()
a = a[a.r != 0].copy()  # exact ties with the old record (mostly unchanged EPS) are neither side
a["rec"] = (a.r > 0).astype(float)
a["sue_pos"], a["sue_neg"] = a.sue_rw.clip(lower=0), a.sue_rw.clip(upper=0)  # asymmetric earnings response
for c in ["size", "mom"]:
    a[c] = a.groupby("ann_mi")[c].transform(lambda s: s.fillna(s.median()))
print("N", len(a), "share records", round(a.rec.mean(), 3))


def rd(df, y, run, h, extra=()):
    d = df[df[run].abs() <= h].dropna(subset=[y, *extra]).copy()
    d["D"] = (d[run] > 0).astype(float)
    d["w"] = 1 - d[run].abs() / h
    d["x"], d["xD"] = d[run], d[run] * d.D
    X = sm.add_constant(d[["D", "x", "xD", *extra]])
    r = sm.WLS(d[y], X, weights=d.w).fit(cov_type="cluster", cov_kwds={"groups": d.ann_mi})
    return r.params.D, r.bse.D, r.tvalues.D, len(d)


rows = []
for run, lab in [("r", "Record"), ("sue_rw", "Beat last year (SUE=0)")]:
    for y in ["ar0", "car1_3"]:
        for h in (0.5, 1.0, 2.0):
            b, se, t, n = rd(a, y, run, h, ("sue_rw",) if run == "r" else ())
            rows.append({"threshold": lab, "outcome": y, "h": h, "jump_pct": 100 * b, "se": 100 * se, "t": t, "n": n})
    for y in ["sue_rw", "size", "mom", "car_pre"] if run == "r" else ["size", "mom", "car_pre"]:
        b, se, t, n = rd(a, y, run, 1.0)
        scale = 100 if y in ("mom", "car_pre") else 1
        rows.append({"threshold": lab, "outcome": f"balance: {y}", "h": 1.0, "jump_pct": scale * b, "se": scale * se, "t": t, "n": n})
R = pd.DataFrame(rows)
print(R.round(3).to_string(index=False))
R.to_csv(OUT / "rd_estimates.csv", index=False)


def density_jump(x, h=2.0, bw=0.05):
    """McCrary-style test: local linear fits to bin counts on each side; jump in log density at 0."""
    edges = np.arange(-h, h + bw / 2, bw)
    cnt, _ = np.histogram(x[(x > -h) & (x <= h)], bins=edges)
    mid = (edges[:-1] + edges[1:]) / 2
    d = pd.DataFrame({"c": cnt, "x": mid, "D": (mid > 0).astype(float)})
    d["xD"] = d.x * d.D
    r = sm.OLS(np.log(d.c.clip(lower=1)), sm.add_constant(d[["D", "x", "xD"]])).fit(cov_type="HC1")
    return r.params.D, r.tvalues.D, d


dj_r, dt_r, dens = density_jump(a.r.to_numpy())
dj_s, dt_s, _ = density_jump(a.sue_rw.to_numpy())
print(f"density jump at record: {100*dj_r:.1f}% (t={dt_r:.2f}); at SUE=0: {100*dj_s:.1f}% (t={dt_s:.2f})")
pd.DataFrame({"threshold": ["Record", "Beat last year"], "log_density_jump": [dj_r, dj_s], "t": [dt_r, dt_s]}).to_csv(
    OUT / "density_tests.csv", index=False)

# Regression kink: does the slope of returns in r change at the record, beyond the surprise?
def kink(df, y, c=0.0, h=1.0):
    d = df[(df.r - c).abs() <= h].dropna(subset=[y, "sue_pos", "sue_neg"]).copy()
    d["x"] = d.r - c
    d["D"] = (d.x > 0).astype(float)
    d["xD"] = d.x * d.D
    d["w"] = 1 - d.x.abs() / h
    X = sm.add_constant(d[["D", "x", "xD", "sue_pos", "sue_neg"]])
    r = sm.WLS(d[y], X, weights=d.w).fit(cov_type="cluster", cov_kwds={"groups": d.ann_mi})
    return r.params.xD, r.bse.xD, r.tvalues.xD, len(d)


krows = []
for y in ["ar0", "car1_3"]:
    for h in (1.0, 2.0):
        b, se, t, n = kink(a, y, 0.0, h)
        krows.append({"outcome": y, "cutoff": 0.0, "h": h, "kink_pct": 100 * b, "se": 100 * se, "t": t, "n": n})
    for c in (-2.0, -1.0, -0.5, 0.5, 1.0):
        b, se, t, n = kink(a, y, c, 1.0)
        krows.append({"outcome": y, "cutoff": c, "h": 1.0, "kink_pct": 100 * b, "se": 100 * se, "t": t, "n": n})
K = pd.DataFrame(krows)
print(K.round(3).to_string(index=False))
K.to_csv(OUT / "kink_estimates.csv", index=False)

# Figure: (a)-(b) returns net of the surprise around the record; (c) kink at true and placebo cutoffs.
d = a[a.r.abs() <= 2].copy()
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55))
for ax, y, title in [(axes[0], "ar0", "(a) Announcement month"), (axes[1], "car1_3", "(b) Next three months")]:
    z = d.dropna(subset=[y, "sue_pos", "sue_neg"])
    X = sm.add_constant(z[["sue_pos", "sue_neg"]])
    z = z.assign(res=sm.OLS(z[y], X).fit().resid + z[y].mean())
    z["bin"] = np.floor(z.r / 0.2) * 0.2 + 0.1
    bm = z.groupby("bin").res.mean()
    ax.scatter(bm.index, 100 * bm.values, s=9, color=SERIES[0], zorder=3)
    for side in (-1, 1):
        s_ = z[np.sign(z.r) == side]
        cf = np.polyfit(s_.r, 100 * s_.res, 1)
        xs = np.linspace(0, 2 * side, 20)
        ax.plot(xs, np.polyval(cf, xs), color=SERIES[1], lw=1.6)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_title(title)
    ax.set_xlabel("Distance to record (SD units)")
axes[0].set_ylabel("Return net of surprise (%)")
kk = K[(K.outcome == "car1_3") & (K.h == 1.0)].sort_values("cutoff")
cols = [SERIES[1] if c == 0 else INK2 for c in kk.cutoff]
for (_, r), col in zip(kk.iterrows(), cols):
    axes[2].errorbar(r.cutoff, r.kink_pct, yerr=1.96 * r.se, fmt="o", color=col, ms=4, capsize=2.5, lw=1)
axes[2].axhline(0, color=INK2, lw=0.8)
axes[2].set_xticks(list(kk.cutoff))
axes[2].set_xticklabels(["-2", "-1", "-.5", "0\n(record)", ".5", "1"])
axes[2].set_title("(c) Drift kink: record vs placebos")
axes[2].set_xlabel("Cutoff (SD units from record)")
axes[2].set_ylabel("Change in slope (%)")
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
print("done idea 3")
