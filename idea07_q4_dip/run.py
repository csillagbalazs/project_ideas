"""Idea 7 PoC: the fourth-quarter earnings dip and whether forecasts and prices anticipate it.

DIP_y = (Q4 EPS - mean(Q1..Q3 EPS)) / sd_ue, measured per firm-year (standalone quarters).
 (a) Size, frequency and persistence of the dip (does last year's dip predict this year's?).
 (b) Do the SRW and XGBoost forecasts anticipate it? (mean scaled error by fiscal quarter)
 (c) Do prices anticipate it? At the annual report, regress the size-adjusted return on the
     XGBoost surprise and on the predictable dip (last year's dip, known in advance) and the
     run-rate surprise a naive extrapolator would perceive.
 (d) Earnings management flavour: the dip by 9-month profitability.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements, winsor  # noqa: E402
from common.stats import fama_macbeth, nw_mean, ols_cluster  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

a = announcements()
w = a.pivot_table(index=["permno", "fyear"], columns="fqtr", values="eps_actual")
w.columns = [f"q{int(c)}" for c in w.columns]
w = w.dropna().reset_index()
q4 = a[a.fqtr == 4].set_index(["permno", "fyear"])
w = w.join(q4[["sd_ue", "sue_rw", "sue_xgb", "e_rw", "e_xgb", "ar0", "car1_3", "ann_mi", "ann_date",
               "size", "bm", "mom", "turnover"]], on=["permno", "fyear"])
w["avg9"] = w[["q1", "q2", "q3"]].mean(axis=1)
w["dip"] = winsor((w.q4 - w.avg9) / w.sd_ue, w.fyear)
w["dip_raw_share"] = (w.q4 < w.avg9).astype(float)
w = w.sort_values(["permno", "fyear"])
w["dip_l1"] = w.groupby("permno").dip.shift(1)
w["dip_l1"] = w.dip_l1.where(w.groupby("permno").fyear.diff() == 1)
w["nine_m"] = w[["q1", "q2", "q3"]].sum(axis=1)
w["loss9"] = (w.nine_m < 0).astype(int)
w = w[w.fyear.between(2004, 2019) & w.dip.notna()]
print("firm-years:", len(w))

# (a) Size, frequency, persistence.
yr = w.groupby("fyear").agg(dip=("dip", "mean"), share=("dip_raw_share", "mean"),
                            rel=("q4", lambda s: np.nan), n=("dip", "size"))
yr["rel"] = w.groupby("fyear").apply(lambda g: ((g.q4 - g.avg9) / g.avg9.abs()).median(), include_groups=False)
print(yr.round(3))
pers = ols_cluster(w, "dip", ["dip_l1"], cluster="fyear")
print("SD dip_l1:", round(w.dip_l1.std(), 3), "IQR", w.dip_l1.quantile([.25,.75]).round(3).tolist())
print("Persistence:", pers.params.round(3).to_dict(), pers.tvalues.round(2).to_dict(), "N", int(pers.nobs))

# (b) Forecast errors by fiscal quarter (scaled by sd_ue).
a2 = a[a.year.between(2005, 2020)].copy()
for m in ["rw", "xgb"]:
    a2[f"se_{m}"] = winsor(a2[f"e_{m}"] / a2.sd_ue, a2.year)
byq = a2.groupby("fqtr")[["se_rw", "se_xgb"]].agg(["mean", "median"])
byq_t = {}
for m in ["se_rw", "se_xgb"]:
    for q in range(1, 5):
        s = a2[a2.fqtr == q].groupby("year")[m].mean()
        byq_t[(m, q)] = nw_mean(s, 2)
print(byq.round(3)); print({k: tuple(round(x, 3) for x in v) for k, v in byq_t.items()})
byq.to_csv(OUT / "errors_by_quarter.csv")

# (c) Prices at the annual report.
w["runrate"] = w.dip  # surprise vs a naive Q4 = average(Q1..Q3) extrapolation
for c in ["size", "bm", "mom", "turnover"]:
    w[c] = w.groupby("ann_mi")[c].transform(lambda s: s.fillna(s.median()))
specs = {
    "xgb": ["sue_xgb"],
    "xgb+lagdip": ["sue_xgb", "dip_l1"],
    "xgb+runrate": ["sue_xgb", "runrate"],
    "all": ["sue_xgb", "runrate", "dip_l1", "size", "bm", "mom", "turnover"],
}
tabs = {}
for dep in ["ar0", "car1_3"]:
    for k, xs in specs.items():
        r = ols_cluster(w, dep, xs, cluster="ann_mi", fe="ann_mi")
        tabs[(dep, k)] = r
        print(dep, k, {v: (round(r.params[v] * 100, 3), round(r.tvalues[v], 2)) for v in xs}, int(r.nobs))

# (d) Dip by 9-month profitability.
grp = w.groupby("loss9").agg(dip=("dip", "mean"), share=("dip_raw_share", "mean"), n=("dip", "size"))
print(grp)
w["small_profit"] = ((w.nine_m > 0) & (w.nine_m < w.groupby("fyear").nine_m.transform(lambda s: s[s > 0].quantile(0.1)))).astype(int)
print(w.groupby("small_profit").dip.mean())

rows = []
for (dep, k), r in tabs.items():
    for v in r.params.index:
        rows.append({"dep": dep, "spec": k, "var": v, "coef": r.params[v], "t": r.tvalues[v], "n": int(r.nobs)})
pd.DataFrame(rows).to_csv(OUT / "price_tests.csv", index=False)
yr.to_csv(OUT / "dip_by_year.csv")
grp.to_csv(OUT / "dip_by_loss.csv")
pd.DataFrame({"coef": pers.params, "t": pers.tvalues, "n": pers.nobs}).to_csv(OUT / "persistence.csv")

# Figure: (left) share of firms with Q4 below the 9M average, by year; (right) mean scaled
# forecast error by fiscal quarter for both models.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
axes[0].plot(yr.index, 100 * yr.share, marker="o", color=SERIES[0], label="Share of firms with Q4 below Q1-Q3 average (%)")
axes[0].plot(yr.index, -100 * yr.rel, marker="o", color=SERIES[1], label="Median Q4 shortfall vs Q1-Q3 average (%)")
axes[0].axhline(50, color=INK2, lw=0.8, ls=":")
axes[0].set_ylabel("%")
axes[0].set_title("The Q4 dip, by fiscal year")
axes[0].set_xticks(range(2004, 2020, 3))
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=7)
x = np.arange(1, 5)
for i, (m, lab) in enumerate([("se_rw", "Seasonal random walk"), ("se_xgb", "XGBoost")]):
    axes[1].bar(x + (i - 0.5) * 0.36, byq[(m, "mean")].values, width=0.34, color=SERIES[i], label=lab)
axes[1].axhline(0, color=INK2, lw=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
axes[1].set_ylabel("Mean error / SD of past errors")
axes[1].set_title("Do the forecasts anticipate Q4?")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7)
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
