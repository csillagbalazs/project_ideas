"""Idea 9 PoC: bundled annual and Q1 reports - do investors neglect one of two signals?

Firm-year classes by the release of the annual report (fiscal Q4 of year y) and the Q1 report
of year y+1:
  same_day   : both on the same day (a bundle)
  same_month : same calendar month, different days
  separate   : different months
Monthly returns cannot split two releases in the same month, so same_month is the control for
bundling at the monthly frequency; separate shows the unbundled benchmark.

Tests (size-adjusted returns, month fixed effects, SEs clustered by month):
  ERC   : release-month return on each surprise, interacted with the class dummies,
          controlling for the other surprise when it falls in the same month.
  Drift : months +1..+3 after the later release, on both surprises x class.
  Selection: who bundles (surprise signs, size), linear probability with year FE.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements  # noqa: E402
from common.stats import ols_cluster  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
SUE = sys.argv[1] if len(sys.argv) > 1 else "sue_xgb"

a = announcements()
keep = ["permno", "fyear", "ann_date", "ann_mi", SUE, "ar0", "car1_3", "size", "bm", "mom", "turnover"]
q4 = a[a.fqtr == 4][keep]
q1 = a[a.fqtr == 1][keep].assign(fyear=lambda d: d.fyear - 1)
w = q4.merge(q1, on=["permno", "fyear"], suffixes=("4", "1"))
w["gap"] = (w.ann_date1 - w.ann_date4).dt.days
w = w[w.gap.between(0, 150) & w.fyear.between(2005, 2019)].copy()
w["cls"] = np.select([w.gap == 0, w.ann_mi1 == w.ann_mi4], ["same_day", "same_month"], "separate")
w = w.dropna(subset=[f"{SUE}4", f"{SUE}1"])
w["sd"] = (w.cls == "same_day").astype(float)
w["sm"] = (w.cls == "same_month").astype(float)
w["joint"] = w.sd + w.sm
w["s4"], w["s1"] = w[f"{SUE}4"], w[f"{SUE}1"]
print(w.cls.value_counts().to_dict())
prev = w.groupby("fyear").cls.value_counts(normalize=True).unstack().fillna(0)

for k in ["4", "1"]:
    for x in ["sd", "sm"]:
        w[f"s{k}_{x}"] = w[f"s{k}"] * w[x]
w["s1_joint"], w["s4_joint"] = w.s1 * w.joint, w.s4 * w.joint
w["later_mi"] = w.ann_mi1  # later release month (same as ann_mi4 for joint classes)

res = {}
# ERC of the annual (Q4) surprise in the annual-report month.
res["erc_q4"] = ols_cluster(w, "ar04", ["s4", "s4_sd", "s4_sm", "s1_joint", "sd", "sm"], cluster="ann_mi4", fe="ann_mi4")
# ERC of the Q1 surprise in the Q1-report month.
res["erc_q1"] = ols_cluster(w, "ar01", ["s1", "s1_sd", "s1_sm", "s4_joint", "sd", "sm"], cluster="ann_mi1", fe="ann_mi1")
# Drift after the later release.
res["drift"] = ols_cluster(w, "car1_31", ["s4", "s4_sd", "s4_sm", "s1", "s1_sd", "s1_sm", "sd", "sm"],
                           cluster="later_mi", fe="later_mi")
# Within the joint-month classes only: does same-day bundling differ from same-month release?
j = w[w.joint == 1]
res["joint_ar0"] = ols_cluster(j, "ar04", ["s4", "s1", "s4_sd", "s1_sd", "sd"], cluster="ann_mi4", fe="ann_mi4")
res["joint_drift"] = ols_cluster(j, "car1_31", ["s4", "s1", "s4_sd", "s1_sd", "sd"], cluster="ann_mi4", fe="ann_mi4")
for k, r in res.items():
    print(k, {v: (round(100 * r.params[v], 3), round(r.tvalues[v], 2)) for v in r.params.index}, int(r.nobs))

# Implied ERCs by class (coef sums) for the figure.
def comb(r, base, inter):
    b = r.params[base] + (r.params[inter] if inter else 0)
    return 100 * b
erc = pd.DataFrame({
    "Annual (Q4) surprise": [comb(res["erc_q4"], "s4", None), comb(res["erc_q4"], "s4", "s4_sm"), comb(res["erc_q4"], "s4", "s4_sd")],
    "Q1 surprise": [comb(res["erc_q1"], "s1", None), comb(res["erc_q1"], "s1", "s1_sm"), comb(res["erc_q1"], "s1", "s1_sd")],
}, index=["Separate months", "Same month", "Same day"])
drift = pd.DataFrame({
    "Annual (Q4) surprise": [comb(res["drift"], "s4", None), comb(res["drift"], "s4", "s4_sm"), comb(res["drift"], "s4", "s4_sd")],
    "Q1 surprise": [comb(res["drift"], "s1", None), comb(res["drift"], "s1", "s1_sm"), comb(res["drift"], "s1", "s1_sd")],
}, index=erc.index)
print(erc.round(3)); print(drift.round(3))

# Selection into same-day bundling.
w["neg4"] = (w.s4 < 0).astype(float)
w["neg1"] = (w.s1 < 0).astype(float)
w["mixed"] = (np.sign(w.s4) != np.sign(w.s1)).astype(float)
w["size4f"] = w.groupby("fyear").size4.transform(lambda s: s.fillna(s.median()))
w["size_z"] = w.groupby("fyear").size4f.transform(lambda s: (s - s.mean()) / s.std())
sel = ols_cluster(w, "sd", ["neg4", "neg1", "mixed", "size_z"], cluster="fyear", fe="fyear")
print("selection:", {v: (round(sel.params[v], 3), round(sel.tvalues[v], 2)) for v in sel.params.index})
# Persistence of the bundling choice (firm habit).
w = w.sort_values(["permno", "fyear"])
w["sd_l1"] = w.groupby("permno").sd.shift(1)
print("P(same day | same day last year) =", round(w[w.sd_l1 == 1].sd.mean(), 3),
      " P(same day | not) =", round(w[w.sd_l1 == 0].sd.mean(), 3))

rows = []
for k, r in {**res, "selection": sel}.items():
    for v in r.params.index:
        rows.append({"model": k, "var": v, "coef": r.params[v], "t": r.tvalues[v], "n": int(r.nobs)})
pd.DataFrame(rows).to_csv(OUT / f"regressions_{SUE}.csv", index=False)
erc.to_csv(OUT / f"erc_by_class_{SUE}.csv")
drift.to_csv(OUT / f"drift_by_class_{SUE}.csv")
prev.to_csv(OUT / "prevalence.csv")

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
for i, (c, lab) in enumerate([("same_day", "Same day"), ("same_month", "Same month"), ("separate", "Separate months")]):
    axes[0].plot(prev.index, 100 * prev[c], marker="o", markersize=3, color=SERIES[i], label=lab)
axes[0].set_title("Release pattern")
axes[0].set_ylabel("% of firm-years")
axes[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
axes[0].set_xticks(range(2005, 2020, 4))
x = np.arange(3)
for ax, tab, title in [(axes[1], erc, "Release-month response"), (axes[2], drift, "Drift, months +1..+3")]:
    for i, col in enumerate(tab.columns):
        ax.bar(x + (i - 0.5) * 0.36, tab[col].values, width=0.34, color=SERIES[i], label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(["Separate", "Same\nmonth", "Same\nday"])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title)
axes[1].set_ylabel("% return per unit SUE")
axes[2].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
fig.tight_layout()
save(fig, OUT / f"fig_main_{SUE}.pdf")
