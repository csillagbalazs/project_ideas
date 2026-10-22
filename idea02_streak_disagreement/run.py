"""Idea 2 PoC: do earnings streaks create disagreement?

If some investors expect streaks to reverse (gambler's fallacy) and others expect them to continue
(hot hand), a streak-continuing surprise should split opinions and raise trading. Rabin and Vayanos
(2010) imply that the split depends on streak length. Outcomes:
  - abnormal turnover in the announcement month and the next month (log turnover minus the firm's
    mean over months -12..-2),
  - change in analyst disagreement: I/B/E/S forecast SD and high-low range, start of m+1 minus start
    of m, scaled by price (May revisions dropped: fiscal-year roll).
Regressors: dummies for each signed streak length (up/down x 2,3,4,5+), with reversals (length 1)
as the base, controlling for |SUE|, |SUE|^2, the sign of the surprise, size and momentum; month FE,
SEs clustered by month. Finally: is the streak drift stronger where turnover (disagreement) is high?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements, monthly_panel, winsor  # noqa: E402
from common.features import abnormal_turnover  # noqa: E402
from common.stats import ols_cluster  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

a = announcements()
a = a[a.year.between(2005, 2020) & a.sue_rw.notna() & (a.streak_rw != 0)].copy()
T = abnormal_turnover()
m = monthly_panel().set_index(["permno", "mi"])


def at(src, k):
    return src.reindex(pd.MultiIndex.from_arrays([a.permno, a.ann_mi + k])).to_numpy()


a["abto0"], a["abto1"] = 100 * at(T, 0), 100 * at(T, 1)
p0 = at(m.price, 0)
a["dsd"] = (at(m.eps_12sd, 1) - at(m.eps_12sd, 0)) / p0
a["drange"] = ((at(m.eps_12high, 1) - at(m.eps_12low, 1)) - (at(m.eps_12high, 0) - at(m.eps_12low, 0))) / p0
a["mon1"] = (a.ann_mi + 1) % 12 + 1
a["asue"] = a.sue_rw.abs()
a["asue2"] = a.asue ** 2
a["pos"] = (a.sue_rw > 0).astype(float)
for c in ["size", "mom"]:
    a[c] = a.groupby("ann_mi")[c].transform(lambda s: s.fillna(s.median()))
a["k"] = a.streak_rw.abs().clip(upper=5)
sign = np.sign(a.streak_rw)
dums = []
for s_, lab in [(1, "up"), (-1, "dn")]:
    for k in (2, 3, 4, 5):
        c = f"{lab}{k}"
        a[c] = ((sign == s_) & (a.k == k)).astype(float)
        dums.append(c)
CTRL = ["asue", "asue2", "pos", "size", "mom"]
X = dums + CTRL

res = {y: ols_cluster(a, y, X, cluster="ann_mi", fe="ann_mi") for y in ["abto0", "abto1"]}
d = a[a.mon1 != 5].copy()
for y in ["dsd", "drange"]:
    d[y] = 1e4 * winsor(d[y], d.ann_mi)  # basis points of price
    res[y] = ols_cluster(d, y, X, cluster="ann_mi", fe="ann_mi")
for y, r in res.items():
    print(y, {v: (round(r.params[v], 2), round(r.tvalues[v], 2)) for v in dums}, int(r.nobs))

# Does disagreement (high abnormal turnover) shape the streak drift?
q = a.dropna(subset=["car1_3", "abto0"]).copy()
q["hi"] = (q.groupby("ann_mi").abto0.rank(pct=True) > 0.5).astype(float)
q["sue"] = q.sue_rw
q["cont"] = (q.k >= 2).astype(float)
q["sue_c"] = q.sue * q.cont
q["sue_hi"] = q.sue * q.hi
q["cont_hi"] = q.cont * q.hi
q["sue_c_hi"] = q.sue_c * q.hi
drift = ols_cluster(q, "car1_3", ["sue", "cont", "sue_c", "hi", "sue_hi", "cont_hi", "sue_c_hi"], cluster="ann_mi", fe="ann_mi")
print("drift", {v: (round(100 * drift.params[v], 3), round(drift.tvalues[v], 2)) for v in drift.params.index})

rows = []
for k, r in {**res, "drift": drift}.items():
    for v in r.params.index:
        rows.append({"model": k, "var": v, "coef": r.params[v], "t": r.tvalues[v], "n": int(r.nobs)})
pd.DataFrame(rows).to_csv(OUT / "regressions.csv", index=False)

# Figure: abnormal turnover by signed streak length relative to a reversal.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
order = ["dn5", "dn4", "dn3", "dn2", "up2", "up3", "up4", "up5"]
labels = ["-5+", "-4", "-3", "-2", "+2", "+3", "+4", "+5+"]
for ax, y, title in [(axes[0], "abto0", "Announcement month"), (axes[1], "abto1", "Following month")]:
    r = res[y]
    b = [r.params[c] for c in order]
    se = [r.bse[c] for c in order]
    cols = [SERIES[1] if c.startswith("dn") else SERIES[0] for c in order]
    ax.bar(range(8), b, color=cols, width=0.65)
    ax.errorbar(range(8), b, yerr=[1.96 * s for s in se], fmt="none", color=INK2, capsize=2.5, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(8))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Streak (sign x length) vs a reversal")
    ax.set_title(title)
axes[0].set_ylabel("Abnormal turnover (%)")
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
print("done idea 2")
