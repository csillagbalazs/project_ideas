"""Idea 11 PoC: waves of inexperienced investors and the processing of earnings news.

New-investor inflow: monthly new A-share investors (CSDC statistics via Eastmoney,
data_raw/eastmoney_new_investors.json, from 2015-04). Inflow_m = log(new investors in m)
minus its trailing 12-month mean (abnormal entry). A longer proxy for 2005-2020 is abnormal
market turnover (log equal-weighted turnover minus its trailing 12-month mean).

Tests: pooled announcement-level regressions with month fixed effects (SEs clustered by month)
    y = b1 SUE + b2 SUE x Inflow_m (+ streak terms)
for the announcement-month response (ar0) and the drift (car1_3). Underreaction amplified by
new investors means b2 < 0 for ar0 and b2 > 0 for drift. Same for the streak interaction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements, monthly_panel  # noqa: E402
from common.stats import ols_cluster  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

raw = json.loads((HERE / "data_raw" / "eastmoney_new_investors.json").read_text())
ni = pd.DataFrame(raw["result"]["data"])[["STATISTICS_DATE", "ADD_INVESTOR", "END_INVESTOR"]]
ni["date"] = pd.to_datetime(ni.STATISTICS_DATE + "-01")
ni["mi"] = ni.date.dt.year * 12 + ni.date.dt.month - 1
ni["lnew"] = np.log(ni.ADD_INVESTOR.astype(float))
ni["inflow"] = ni.lnew - ni.lnew.rolling(12, min_periods=6).mean().shift(1)
ni = ni[["mi", "date", "ADD_INVESTOR", "inflow"]]

m = monthly_panel()
# Row m+1 holds month-m turnover; shift back so turnover is indexed by the month it was traded.
mt = m.groupby("mi").turnover.mean().rename("to_ew").to_frame()
mt.index = mt.index - 1
mt["lto"] = np.log(mt.to_ew)
mt["to_abn"] = mt.lto - mt.lto.rolling(12, min_periods=6).mean().shift(1)
mkt = m.groupby("mi").mkt_vw.first()
series = mt.join(ni.set_index("mi"), how="left").join(mkt)
ov = series.dropna(subset=["inflow", "to_abn"])
corr = ov[["inflow", "to_abn"]].corr().iloc[0, 1]
print(f"overlap months {len(ov)}; corr(abnormal new investors, abnormal turnover) = {corr:.2f}")

a = announcements()
a = a[a.year.between(2005, 2020) & a.sue_xgb.notna() & a.ar0.notna()].copy()
z = lambda s: (s - s.mean()) / s.std()
a["sue"] = a.groupby("ann_mi").sue_xgb.transform(z)
a["sue_rw_z"] = a.groupby("ann_mi").sue_rw.transform(z)
a["cont"] = (a.streak_rw.abs() >= 2).astype(float)
a = a.join(series[["inflow", "to_abn", "mkt_vw"]], on="ann_mi")
for c in ["inflow", "to_abn"]:
    s = series[c].dropna()
    a[c + "_z"] = (a[c] - s.mean()) / s.std()

res = {}
for proxy, sub in [("inflow_z", a[a.inflow.notna()]), ("to_abn_z", a[a.to_abn.notna()])]:
    sub = sub.copy()
    sub["sue_x"] = sub.sue * sub[proxy]
    sub["sue_rw_x_cont"] = sub.sue_rw_z * sub.cont
    sub["sue_rw_x_cont_x"] = sub.sue_rw_x_cont * sub[proxy]
    sub["sue_rw_x"] = sub.sue_rw_z * sub[proxy]
    sub["cont_x"] = sub.cont * sub[proxy]
    for dep in ["ar0", "car1_3"]:
        r = ols_cluster(sub, dep, ["sue", "sue_x"], cluster="ann_mi", fe="ann_mi")
        res[(proxy, dep, "sue")] = r
        r2 = ols_cluster(sub, dep, ["sue_rw_z", "cont", "sue_rw_x_cont", "sue_rw_x", "cont_x", "sue_rw_x_cont_x"],
                         cluster="ann_mi", fe="ann_mi")
        res[(proxy, dep, "streak")] = r2
    print(proxy, "N", len(sub), "months", sub.ann_mi.nunique(), sub.ann_date.min().date(), sub.ann_date.max().date())
rows = []
for (proxy, dep, spec), r in res.items():
    print(proxy, dep, spec, {v: (round(100 * r.params[v], 3), round(r.tvalues[v], 2)) for v in r.params.index})
    for v in r.params.index:
        rows.append({"proxy": proxy, "dep": dep, "spec": spec, "var": v, "coef": r.params[v], "t": r.tvalues[v], "n": int(r.nobs)})
pd.DataFrame(rows).to_csv(OUT / "regressions.csv", index=False)
series.to_csv(OUT / "monthly_series.csv")

# Split-sample illustration: drift slope in high- vs low-inflow months.
def slopes(sub, dep):
    out = []
    for mi_, g in sub.groupby("ann_mi"):
        if len(g) >= 50:
            out.append((mi_, np.polyfit(g.sue, g[dep], 1)[0]))
    return pd.DataFrame(out, columns=["ann_mi", "b"]).set_index("ann_mi").b
sl = pd.DataFrame({"erc": slopes(a, "ar0"), "drift": slopes(a.dropna(subset=["car1_3"]), "car1_3")})
sl = sl.join(series[["inflow", "to_abn"]])
for proxy in ["inflow", "to_abn"]:
    s = sl.dropna(subset=[proxy])
    hi = s[proxy] > s[proxy].median()
    print(proxy, "months", len(s), "ERC hi/lo", round(100 * s[hi].erc.mean(), 3), round(100 * s[~hi].erc.mean(), 3),
          "| drift hi/lo", round(100 * s[hi].drift.mean(), 3), round(100 * s[~hi].drift.mean(), 3))
sl.to_csv(OUT / "monthly_slopes.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax = axes[0]
s = series.dropna(subset=["ADD_INVESTOR"])
s = s[s.index <= 2021 * 12 + 3]
ax.plot(pd.to_datetime(s.date), s.ADD_INVESTOR / 100, color=SERIES[0], label="New investors (millions)")
ax.set_title("Monthly new A-share investors (CSDC)")
ax.set_ylabel("Millions")
ax2_vals = s.to_abn
for proxy, lab, col, k in [("inflow", "New-investor inflow", SERIES[0], 0), ("to_abn", "Abnormal turnover", SERIES[1], 1)]:
    r_e, r_d = res[(proxy + "_z", "ar0", "sue")], res[(proxy + "_z", "car1_3", "sue")]
    vals = [100 * r_e.params["sue_x"], 100 * r_d.params["sue_x"]]
    ts = [r_e.tvalues["sue_x"], r_d.tvalues["sue_x"]]
    xs = np.arange(2) + (k - 0.5) * 0.36
    axes[1].bar(xs, vals, width=0.34, color=col, label=lab)
    for x_, v_, t_ in zip(xs, vals, ts):
        axes[1].text(x_, v_ + (0.02 if v_ >= 0 else -0.06), f"t={t_:.1f}", ha="center", fontsize=6.5, color=INK2)
axes[1].axhline(0, color=INK2, lw=0.8)
lo, hi = axes[1].get_ylim()
axes[1].set_ylim(lo - 0.08, hi + 0.05)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Announcement month", "Drift, months +1..+3"])
axes[1].set_ylabel("SUE x proxy (% per 1 SD)")
axes[1].set_title("Does investor entry change the response?")
axes[1].legend(fontsize=7, loc="upper left")
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
