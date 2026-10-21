"""Idea 6 PoC: sticky analyst expectations and the profitability anomaly in China.

Following Bouchaud, Krueger, Landier and Thesmar (2019):
 (a) Coibion-Gorodnichenko at the consensus level: FE_m = R_m - F_m on the 3-month revision.
     A positive slope means the consensus underreacts (sticky expectations).
 (b) FE on profitability: a positive slope means analysts are too pessimistic about
     profitable firms.
 (c) The profitability anomaly: next-month returns on ROE.
 (d) Stickiness proxy, firm level, past data only: autocorrelation of the firm's monthly
     consensus revisions over the previous 24 months. Sticky updating spreads news over
     several revisions, so revisions are positively autocorrelated. Prediction: the
     anomaly is stronger where expectations are stickier.
ROE = I/B/E/S trailing EPS / book value per share (both on the share basis of their date).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import factors, winsor  # noqa: E402
from common.features import analyst_panel  # noqa: E402
from common.stats import alpha, fama_macbeth, nw_mean, sort_portfolios  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

d = analyst_panel()
d = d[d.mi >= 2005 * 12].copy()
# Main ROE: realised TTM EPS (as-of share basis) / book value per share. The I/B/E/S
# trailing field blends in forecasts, so it is kept only as a comparison.
d["roe"] = (d.ttm_asof / d.bookvalue).where(d.bookvalue > 0)
d["roe_ibes"] = (d.eps_12trl / d.bookvalue).where(d.bookvalue > 0)
for c in ["FE", "rev1", "rev3", "roe", "roe_ibes", "F"]:
    d[c] = winsor(d[c], d.mi)

# Stickiness proxy: rolling 24-month autocorrelation of revisions, known at the start of m.
d = d.sort_values(["permno", "mi"])
def roll_ac(s):
    return s.rolling(24, min_periods=18).apply(lambda x: pd.Series(x).autocorr(1), raw=True)
d["rev1_l"] = d.groupby("permno").rev1.shift(1)  # revision up to m-1 only
d["stick"] = d.groupby("permno", group_keys=False).rev1_l.apply(roll_ac)
print("stickiness proxy: mean", d.stick.mean().round(3), "share>0", (d.stick > 0).mean().round(3),
      "N", d.stick.notna().sum())

z = lambda s: (s - s.mean()) / s.std()
for c in ["roe", "roe_ibes", "rev3", "stick", "size", "bm", "mom", "rev", "turnover"]:
    d[c + "_z"] = d.groupby("mi")[c].transform(z)

res = {}
# (a) CG regression and (b) FE on ROE; FE needs the realised outcome.
e = d.dropna(subset=["FE"]).copy()
res["cg"] = fama_macbeth(e, "FE", ["rev3"], t="mi", lags=12)
res["fe_roe"] = fama_macbeth(e, "FE", ["roe"], t="mi", lags=12)
res["fe_both"] = fama_macbeth(e, "FE", ["rev3", "roe"], t="mi", lags=12)
e["hi_stick"] = (e.groupby("mi").stick.rank(pct=True) > 0.5).astype(float)
for k, g in e.dropna(subset=["stick"]).groupby("hi_stick"):
    res[f"cg_stick{int(k)}"] = fama_macbeth(g, "FE", ["rev3"], t="mi", lags=12)
    res[f"fe_roe_stick{int(k)}"] = fama_macbeth(g, "FE", ["roe"], t="mi", lags=12)
for k, v in res.items():
    print(k, v.round(4).to_dict("index"))

# (c) Profitability anomaly and (d) interaction with stickiness.
ctrl = ["size_z", "bm_z", "mom_z", "rev_z", "turnover_z"]
fm_roe = fama_macbeth(d, "ret", ["roe_z", *ctrl], t="mi")
fm_ibes = fama_macbeth(d, "ret", ["roe_ibes_z", *ctrl], t="mi")
fm_both = fama_macbeth(d, "ret", ["roe_z", "roe_ibes_z", *ctrl], t="mi")
print("I/B/E/S ROE:", fm_ibes.round(4).loc["roe_ibes_z"].to_dict(), "both:", fm_both.round(4).loc[["roe_z", "roe_ibes_z"]].to_dict("index"))
res["fe_roe_ibes"] = fama_macbeth(d.dropna(subset=["FE"]), "FE", ["roe_ibes"], t="mi", lags=12)
d["roe_x_stick"] = d.roe_z * d.stick_z
fm_int = fama_macbeth(d, "ret", ["roe_z", "stick_z", "roe_x_stick", *ctrl], t="mi")
print(fm_roe.round(4)); print(fm_int.round(4))

fac = factors()
port = {}
p_all = sort_portfolios(d, "roe", "ret", n=5)
port["All"] = p_all.HL
d["stick_grp"] = d.groupby("mi").stick.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)
                                                 if s.notna().sum() > 60 else np.nan)
for gk, lab in [(0, "Low stickiness"), (2, "High stickiness")]:
    port[lab] = sort_portfolios(d[d.stick_grp == gk], "roe", "ret", n=5).HL
rows = {}
for lab, s in port.items():
    m_, t_ = nw_mean(s)
    a4, t4, _ = alpha(s, fac, "CH4")
    a3, t3, _ = alpha(s, fac, "CH3")
    rows[lab] = {"HL": m_, "t": t_, "CH3": a3, "tCH3": t3, "CH4": a4, "tCH4": t4, "months": s.notna().sum()}
diff = (port["High stickiness"] - port["Low stickiness"]).dropna()
dm, dt = nw_mean(diff)
rows["High - Low"] = {"HL": dm, "t": dt}
ports = pd.DataFrame(rows).T
print(ports.round(4))
ports.to_csv(OUT / "roe_portfolios.csv")
pd.concat({k: pd.DataFrame(v[["coef", "t"]].to_dict()) for k, v in res.items()}).to_csv(OUT / "fe_regressions.csv")
pd.concat({"roe": pd.DataFrame(fm_roe[["coef", "t"]].to_dict()),
           "roe_ibes": pd.DataFrame(fm_ibes[["coef", "t"]].to_dict()), "both": pd.DataFrame(fm_both[["coef", "t"]].to_dict()),
           "interaction": pd.DataFrame(fm_int[["coef", "t"]].to_dict())}).to_csv(OUT / "return_fm.csv")

# Figure: (left) mean FE by ROE decile; (right) ROE quintile returns by stickiness group.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
e["roe_dec"] = np.ceil(10 * e.groupby("mi").roe.rank(pct=True)).clip(1, 10)
b = e.groupby("roe_dec").FE.mean()
axes[0].plot(b.index, 100 * b.values, marker="o", color=SERIES[0])
axes[0].axhline(0, color=INK2, lw=0.8)
axes[0].set_xticks(range(1, 11))
axes[0].set_xlabel("ROE decile")
axes[0].set_ylabel("Realised minus forecast (% of price)")
axes[0].set_title("Analyst error by profitability")
for gk, lab, c in [(0, "Low stickiness", SERIES[1]), (2, "High stickiness", SERIES[0])]:
    p = sort_portfolios(d[d.stick_grp == gk], "roe", "ret", n=5)
    axes[1].plot(range(1, 6), 100 * p[[1, 2, 3, 4, 5]].mean().values, marker="o", color=c, label=lab)
axes[1].set_xticks(range(1, 6))
axes[1].set_xlabel("ROE quintile")
axes[1].set_ylabel("Next-month return (%, EW)")
axes[1].set_title("Profitability premium by stickiness")
axes[1].legend(loc="best")
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
