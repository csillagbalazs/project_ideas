"""Idea 10 PoC: the disposition effect and the speed at which earnings news enters prices.

Capital-gains overhang (Grinblatt and Han, 2005) at the start of month m:
    RP_m = sum_n w_n P_{m-n},  w_n = V_{m-n} * prod_{k<n} (1 - V_{m-k}), normalised,
    CGO_m = (P_m - RP_m) / P_m,
with monthly turnover V (capped at 1) and prices put on a common share basis with the
share-event factor (36-month window). Missing turnover (suspensions) counts as no trading. Row j of the panel holds the end-of-month-(j-1) price
and the month-(j-1) turnover, so a window of rows j <= m uses only information known at m.

Frazzini (2006): drift is strongest when news and the overhang share a sign. Test with
Fama-MacBeth regressions of months +1..+3 abnormal returns on SUE, CGO and SUE x CGO, and a
3x3 double sort. Then check whether CGO accounts for the streak result of the
gambler's-fallacy paper (SRW streaks, continuation vs reversal).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements, monthly_panel  # noqa: E402
from common.stats import fama_macbeth, nw_mean  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
N = 36

m = monthly_panel()[["permno", "mi", "price", "turnover", "basis_factor"]].sort_values(["permno", "mi"])
m["padj"] = m.price / m.basis_factor
m["padj"] = m.groupby("permno").padj.ffill()          # suspended months keep the last price
m["V"] = m.turnover.clip(0, 1).fillna(0.0)           # and count as months without trading
cgo = np.full(len(m), np.nan)
for _, idx in m.groupby("permno", sort=False).indices.items():
    p, v, mi = m.padj.to_numpy()[idx], m.V.to_numpy()[idx], m.mi.to_numpy()[idx]
    for j in range(N, len(idx)):
        if mi[j] - mi[j - N + 1] != N - 1:
            continue
        pw, vw = p[j - N + 1:j + 1][::-1], v[j - N + 1:j + 1][::-1]  # n = 0 is the latest row
        if np.isnan(pw).any() or np.isnan(vw).any():
            continue
        surv = np.concatenate([[1.0], np.cumprod(1 - vw[:-1])])
        w = vw * surv
        if w.sum() <= 0:
            continue
        rp = (w * pw).sum() / w.sum()
        cgo[idx[j]] = (p[j] - rp) / p[j]
m["cgo"] = cgo
m["cgo"] = m.cgo.clip(m.cgo.quantile(0.01), m.cgo.quantile(0.99))
print("CGO coverage:", m.cgo.notna().sum(), "mean", round(m.cgo.mean(), 3), "median", round(m.cgo.median(), 3))

a = announcements()
a = a.merge(m[["permno", "mi", "cgo"]].rename(columns={"mi": "ann_mi"}), on=["permno", "ann_mi"], how="left")
a = a[a.year.between(2005, 2020) & a.sue_xgb.notna() & a.cgo.notna() & a.car1_3.notna()].copy()
for c in ["size", "bm", "mom", "turnover"]:
    a[c] = a.groupby("ann_mi")[c].transform(lambda s: s.fillna(s.median()))
z = lambda s: (s - s.mean()) / s.std()
for c in ["sue_xgb", "sue_rw", "cgo", "size", "bm", "mom", "turnover"]:
    a[c + "_z"] = a.groupby("ann_mi")[c].transform(z)
a["sue_x_cgo"] = a.sue_xgb_z * a.cgo_z
a["aligned"] = (np.sign(a.sue_xgb) == np.sign(a.cgo)).astype(float)
print("N:", len(a), "corr(SUE, CGO) =", round(a[["sue_xgb", "cgo"]].corr().iloc[0, 1], 3))
ctrl = ["size_z", "bm_z", "mom_z", "turnover_z"]

res = {}
for dep in ["ar0", "car1_3"]:
    res[(dep, "base")] = fama_macbeth(a, dep, ["sue_xgb_z", "cgo_z", "sue_x_cgo"], t="ann_mi")
    res[(dep, "ctrl")] = fama_macbeth(a, dep, ["sue_xgb_z", "cgo_z", "sue_x_cgo", *ctrl], t="ann_mi")
for k, r in res.items():
    print(k, {v: (round(100 * r.loc[v, "coef"], 3), round(r.loc[v, "t"], 2)) for v in r.index})

# Double sort: SUE terciles x CGO terciles within announcement month.
a["sue_t"] = np.ceil(3 * a.groupby("ann_mi").sue_xgb.rank(pct=True)).clip(1, 3)
a["cgo_t"] = np.ceil(3 * a.groupby("ann_mi").cgo.rank(pct=True)).clip(1, 3)
big = a.groupby("ann_mi").ann_mi.transform("size") >= 300  # main reporting months, >= ~33 per cell
print("double-sort months:", a[big].ann_mi.nunique(), "announcements:", int(big.sum()))
ds = a[big].groupby(["ann_mi", "sue_t", "cgo_t"]).car1_3.mean().unstack(["sue_t", "cgo_t"])
grid = ds.mean().unstack("cgo_t")
spread = {c: nw_mean(ds[(3.0, c)] - ds[(1.0, c)]) for c in [1.0, 2.0, 3.0]}
# Frazzini's aligned strategy: long good news/high gains, short bad news/high losses.
hl_diff = nw_mean((ds[(3.0, 3.0)] - ds[(1.0, 3.0)]) - (ds[(3.0, 1.0)] - ds[(1.0, 1.0)]))
print("PEAD spread high CGO minus low CGO:", round(100 * hl_diff[0], 2), "t =", round(hl_diff[1], 2))
gn = nw_mean(ds[(3.0, 3.0)] - ds[(3.0, 1.0)]); bn = nw_mean(ds[(1.0, 1.0)] - ds[(1.0, 3.0)])
print("good news: gains minus losses", round(100 * gn[0], 2), round(gn[1], 2), "| bad news: losses minus gains", round(100 * bn[0], 2), round(bn[1], 2))
fz = ds[(3.0, 3.0)] - ds[(1.0, 1.0)]
fz_m, fz_t = nw_mean(fz)
mis = ds[(3.0, 1.0)] - ds[(1.0, 3.0)]
mis_m, mis_t = nw_mean(mis)
d_m, d_t = nw_mean(fz - mis)
print(grid.round(4))
print("SUE spread by CGO tercile:", {k: (round(100 * v[0], 2), round(v[1], 2)) for k, v in spread.items()})
print(f"aligned: {100*fz_m:.2f} (t={fz_t:.2f}); misaligned: {100*mis_m:.2f} (t={mis_t:.2f}); diff {100*d_m:.2f} (t={d_t:.2f})")

# Streak interplay (SRW surprises, as in the gambler's-fallacy paper).
a["cont"] = (a.streak_rw.abs() >= 2).astype(float)
a["sue_rw_x_cont"] = a.sue_rw_z * a.cont
a["sue_rw_x_cgo"] = a.sue_rw_z * a.cgo_z
a["cgo_signed"] = a.cgo * np.sign(a.streak_rw)  # gains on up-streak / losses on down-streak
print("mean signed CGO: continuation", round(a[a.cont == 1].cgo_signed.mean(), 3),
      " reversal", round(a[a.cont == 0].cgo_signed.mean(), 3))
st1 = fama_macbeth(a, "car1_3", ["sue_rw_z", "cont", "sue_rw_x_cont", *ctrl], t="ann_mi")
st2 = fama_macbeth(a, "car1_3", ["sue_rw_z", "cont", "sue_rw_x_cont", "cgo_z", "sue_rw_x_cgo", *ctrl], t="ann_mi")
for lab, r in [("streak", st1), ("streak+cgo", st2)]:
    print(lab, {v: (round(100 * r.loc[v, "coef"], 3), round(r.loc[v, "t"], 2)) for v in r.index if not v.endswith("_z") or v.startswith("sue") or v.startswith("cgo")})

rows = []
for k, r in {**{f"{a_}_{b_}": v for (a_, b_), v in res.items()}, "streak": st1, "streak_cgo": st2}.items():
    for v in r.index:
        rows.append({"model": k, "var": v, "coef": r.loc[v, "coef"], "t": r.loc[v, "t"]})
pd.DataFrame(rows).to_csv(OUT / "fama_macbeth.csv", index=False)
grid.to_csv(OUT / "double_sort_car1_3.csv")
pd.DataFrame({"aligned": [fz_m, fz_t], "misaligned": [mis_m, mis_t], "diff": [d_m, d_t]}, index=["mean", "t"]).to_csv(OUT / "frazzini_strategy.csv")

# Figure: (left) drift by SUE tercile for low and high CGO; (right) SUE drift spread by CGO tercile.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for c, lab, col in [(1.0, "Capital losses (low CGO)", SERIES[1]), (3.0, "Capital gains (high CGO)", SERIES[0])]:
    axes[0].plot([1, 2, 3], 100 * grid[c].values, marker="o", color=col, label=lab)
axes[0].set_xticks([1, 2, 3])
axes[0].set_xticklabels(["Bad news", "Neutral", "Good news"])
axes[0].set_ylabel("Abn. return, months +1..+3 (%)")
axes[0].set_title("Drift by news and overhang")
axes[0].axhline(0, color=INK2, lw=0.8)
axes[0].legend(fontsize=7, loc="upper left")
vals = [100 * spread[c][0] for c in [1.0, 2.0, 3.0]]
axes[1].bar([1, 2, 3], vals, color=SERIES[0], width=0.6)
for i, c in enumerate([1.0, 2.0, 3.0]):
    axes[1].text(i + 1, vals[i] + 0.05, f"t={spread[c][1]:.1f}", ha="center", fontsize=7, color=INK2)
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(["Low CGO", "Mid", "High CGO"])
axes[1].set_ylabel("Good minus bad news drift (%)")
axes[1].set_title("PEAD spread by overhang tercile")
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
