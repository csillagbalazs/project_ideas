"""Idea 1 PoC: do sell-side analysts commit the gambler's fallacy?

For each quarterly announcement we measure, inside I/B/E/S (same share basis as prices):
  revision  REV = (consensus at start of month m+1 - consensus at start of month m) / price_m
  error     FE  = (I/B/E/S trailing EPS twelve months after m+1 - consensus at m+1) / price_m
where m is the announcement month. A positive FE slope on SUE means the post-announcement
consensus still underreacts to the surprise.

Streaks follow the gambler's-fallacy paper: consecutive SRW surprises of the same sign. A
surprise with |streak| >= 2 continues a streak; |streak| = 1 reverses (or starts) one.
Gambler's fallacy: analysts expect streaks to end, so they revise LESS per unit of surprise
after a continuation and their errors show MORE underreaction after continuations.
Months whose revision spans the I/B/E/S fiscal-year roll (May) are dropped, because the
12-month-forward blend jumps mechanically then.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements, monthly_panel, winsor  # noqa: E402
from common.stats import ols_cluster  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

a = announcements()
m = monthly_panel().set_index(["permno", "mi"])


def at(df, k, col):
    return m[col].reindex(pd.MultiIndex.from_arrays([df.permno, df.ann_mi + k])).to_numpy()


a = a[a.year.between(2005, 2020)].copy()
a["fwd0"], a["fwd1"] = at(a, 0, "eps_12fwd"), at(a, 1, "eps_12fwd")
a["trl0"], a["p0"], a["trl13"] = at(a, 0, "eps_12trl"), at(a, 0, "price"), at(a, 13, "eps_12trl")
a["rev"] = (a.fwd1 - a.fwd0) / a.p0
a["fe"] = (a.trl13 - a.fwd1) / a.p0
a["fg"] = (a.fwd0 - a.trl0) / a.p0  # expected growth: absorbs the mechanical drift of the time-weighted blend
a["mon1"] = (a.ann_mi + 1) % 12 + 1

# XGBoost-based streaks (surprise signs relative to the ML forecast), for robustness.
a = a.sort_values(["permno", "fyear", "fqtr"])
sx = np.zeros(len(a))
for _, idx in a.groupby("permno", sort=False).indices.items():
    run = 0
    for j, v in zip(idx, np.sign(a.e_xgb.to_numpy()[idx])):
        run = 0 if (np.isnan(v) or v == 0) else (run + v if run != 0 and np.sign(run) == v else v)
        sx[j] = run
a["streak_xgb"] = sx


def prep(df, sue, streak):
    d = df.copy()
    d["sue"] = d[sue]
    d["k"] = d[streak].abs().clip(upper=4)
    d["cont"] = (d.k >= 2).astype(float)
    d["sue_c"] = d.sue * d.cont
    for k in (1, 2, 3, 4):
        d[f"L{k}"] = (d.k == k).astype(float)
        d[f"sueL{k}"] = d.sue * d[f"L{k}"]
    return d


b = a.dropna(subset=["rev", "fg", "sue_rw"])
b = b[b.mon1 != 5].copy()
for c in ["rev", "fe", "fg"]:
    b[c] = winsor(b[c], b.ann_mi)
B = prep(b[b.streak_rw != 0], "sue_rw", "streak_rw")
BX = prep(b.dropna(subset=["sue_xgb"])[lambda d: d.streak_xgb != 0], "sue_xgb", "streak_xgb")
print("N revisions:", len(B), "with FE:", B.fe.notna().sum(), "share continuations:", B.cont.mean().round(3))

X1 = ["sue", "cont", "sue_c", "fg"]
X2 = ["sueL1", "sueL2", "sueL3", "sueL4", "L2", "L3", "L4", "fg"]
res = {
    "rev": ols_cluster(B, "rev", X1, cluster="ann_mi", fe="ann_mi"),
    "fe": ols_cluster(B, "fe", X1, cluster="ann_mi", fe="ann_mi"),
    "rev_len": ols_cluster(B, "rev", X2, cluster="ann_mi", fe="ann_mi"),
    "fe_len": ols_cluster(B, "fe", X2, cluster="ann_mi", fe="ann_mi"),
    "rev_xgb": ols_cluster(BX, "rev", X1, cluster="ann_mi", fe="ann_mi"),
    "fe_xgb": ols_cluster(BX, "fe", X1, cluster="ann_mi", fe="ann_mi"),
}
for k, r in res.items():
    print(k, {v: (round(100 * r.params[v], 4), round(r.tvalues[v], 2)) for v in r.params.index if v != "fg"}, int(r.nobs))

# Price link: does the post-announcement drift load more on streak-continuing surprises, and is
# it concentrated in analyst-covered firms (where analysts could transmit the bias)?
P = prep(a[a.streak_rw != 0].dropna(subset=["car1_3", "sue_rw"]), "sue_rw", "streak_rw")
P["covered"] = P.fwd0.notna()
price = {lab: ols_cluster(sub, "car1_3", ["sue", "cont", "sue_c"], cluster="ann_mi", fe="ann_mi")
         for lab, sub in [("All", P), ("Covered", P[P.covered]), ("Not covered", P[~P.covered])]}
for k, r in price.items():
    print("drift", k, {v: (round(100 * r.params[v], 3), round(r.tvalues[v], 2)) for v in ["sue", "sue_c"]}, int(r.nobs))

rows = []
for k, r in {**res, **{f"drift_{k}": v for k, v in price.items()}}.items():
    for v in r.params.index:
        rows.append({"model": k, "var": v, "coef": r.params[v], "t": r.tvalues[v], "n": int(r.nobs)})
pd.DataFrame(rows).to_csv(OUT / "regressions.csv", index=False)

# Figure: slope of revision and of later error on SUE, by streak length.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
lab = ["1\n(reversal)", "2", "3", "4+"]
for ax, key, title, ylab in [(axes[0], "rev_len", "Consensus revision per unit of surprise", "x 100 (% of price)"),
                             (axes[1], "fe_len", "Later forecast error per unit of surprise", "x 100 (% of price)")]:
    r = res[key]
    b_ = [100 * r.params[f"sueL{k}"] for k in (1, 2, 3, 4)]
    se = [100 * r.bse[f"sueL{k}"] for k in (1, 2, 3, 4)]
    ax.bar(range(4), b_, color=SERIES[0], width=0.6)
    ax.errorbar(range(4), b_, yerr=[1.96 * s for s in se], fmt="none", color=INK2, capsize=3, lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lab)
    ax.set_xlabel("Streak length including this surprise")
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.axhline(0, color=INK2, lw=0.8)
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
print("done idea 1")
