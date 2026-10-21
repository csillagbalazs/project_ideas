"""Idea 5 PoC: conditional bias of Chinese analysts against a look-ahead-free ML benchmark.

Target: R_m = trailing I/B/E/S EPS twelve months later / price at m.
Analyst forecast: F_m = 12-month-forward consensus / price.
Machine forecasts (XGBoost, retrained every January on rows whose outcome was already
public, i.e. m + 12 < forecast month):
  ML-noA : public accounting and market data only
  ML+A   : the same plus the consensus (``man + machine'')
Conditional bias = F - ML-noA. Portfolios are formed at the start of month m and earn the
return of month m.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import factors, winsor  # noqa: E402
from common.features import analyst_panel  # noqa: E402
from common.stats import alpha, fama_macbeth, nw_mean, sort_portfolios  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

d = analyst_panel()
d = d[d.mi >= 2005 * 12].copy()
d["month"] = d.mi % 12 + 1
for c in ["F", "T", "R", "disp", "range", "rev1", "rev3", "fgrowth", "bp"]:
    d[c] = winsor(d[c], d.mi)
BASE = ["T", "bp", "size", "mom", "rev", "turnover", "ttm_growth", "pos_share4", "sue_xgb", "sue_rw",
        "streak_rw", "roe_sign", "month", "fqtr"]
ANALYST = ["F", "disp", "range", "rev1", "rev3", "fgrowth"]
PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=50, objective="reg:absoluteerror", n_jobs=8, random_state=0)

preds = []
for Y in range(2009, 2021):
    test = d[(d.mi >= Y * 12) & (d.mi < (Y + 1) * 12)]
    train = d[(d.mi + 12 < Y * 12) & d.R.notna()]
    row = {"year": Y, "n_train": len(train), "n_test": len(test)}
    out = test[["permno", "mi"]].copy()
    for name, feats in [("ml_noA", BASE), ("ml_A", BASE + ANALYST)]:
        mdl = xgb.XGBRegressor(**PARAMS).fit(train[feats], train.R)
        out[name] = mdl.predict(test[feats])
    preds.append(out)
    print(row)
d = d.merge(pd.concat(preds), on=["permno", "mi"], how="inner")
d["bias"] = d.F - d.ml_noA
print("Test rows:", len(d), "months:", d.mi.nunique(), "firms:", d.permno.nunique())

# Accuracy on rows with realised outcomes.
e = d.dropna(subset=["R"])
acc = pd.DataFrame({
    "MAE": [(e.R - e.F).abs().mean(), (e.R - e.ml_noA).abs().mean(), (e.R - e.ml_A).abs().mean()],
    "Mean error (R - f)": [(e.R - e.F).mean(), (e.R - e.ml_noA).mean(), (e.R - e.ml_A).mean()],
    "Share better than analysts": [np.nan, ((e.R - e.ml_noA).abs() < (e.R - e.F).abs()).mean(),
                                   ((e.R - e.ml_A).abs() < (e.R - e.F).abs()).mean()],
}, index=["Analysts (consensus)", "ML without analysts", "ML with analysts"])
print(acc)
acc.to_csv(OUT / "accuracy.csv")
by_year = e.assign(year=e.mi // 12).groupby("year").apply(
    lambda g: pd.Series({"analyst": (g.R - g.F).abs().mean(), "ml_noA": (g.R - g.ml_noA).abs().mean(),
                         "ml_A": (g.R - g.ml_A).abs().mean(), "bias": (g.F - g.R).mean()}), include_groups=False)
print(by_year.round(4))
by_year.to_csv(OUT / "accuracy_by_year.csv")

# Does the conditional bias predict the analyst error and returns?
fe = fama_macbeth(e, "FE", ["bias"], t="mi")
print("FE on bias:", fe.round(4).to_dict("index"))
d["ret_pct"] = d.ret
fac = factors()
rf = fac.set_index("mi").rf_mon
res = {}
for w in [None, "me"]:
    p = sort_portfolios(d, "bias", "ret", n=5, t="mi", weight=w)
    lab = "VW" if w else "EW"
    hl_m, hl_t = nw_mean(p.HL)
    a4, t4, _ = alpha(p.HL, fac, "CH4")
    a3, t3, _ = alpha(p.HL, fac, "CH3")
    res[lab] = {"Q1": p[1].mean(), "Q3": p[3].mean(), "Q5": p[5].mean(), "HL": hl_m, "t": hl_t,
                "CH3": a3, "tCH3": t3, "CH4": a4, "tCH4": t4}
    if lab == "EW":
        p_ew = p
res = pd.DataFrame(res).T
print(res.round(4))
res.to_csv(OUT / "portfolios.csv")

for c in ["bias", "F", "ml_noA", "size", "bm", "mom", "rev", "turnover"]:
    d[c + "_z"] = d.groupby("mi")[c].transform(lambda s: (s - s.mean()) / s.std())
fm1 = fama_macbeth(d, "ret", ["bias_z", "size_z", "bm_z", "mom_z", "rev_z", "turnover_z"], t="mi")
fm2 = fama_macbeth(d, "ret", ["F_z", "ml_noA_z", "size_z", "bm_z", "mom_z", "rev_z", "turnover_z"], t="mi")
print(fm1.round(4)); print(fm2.round(4))
pd.concat({"bias": pd.DataFrame(fm1[["coef", "t"]].to_dict()), "split": pd.DataFrame(fm2[["coef", "t"]].to_dict())}).to_csv(OUT / "fama_macbeth.csv")

# Subperiods for the HL spread.
hl = p_ew.HL
sub = {f"{a}-{b}": nw_mean(hl[(hl.index >= a * 12) & (hl.index < (b + 1) * 12)]) for a, b in [(2009, 2014), (2015, 2020)]}
print("Subperiods EW HL:", sub)
pd.DataFrame(sub, index=["mean", "t"]).to_csv(OUT / "subperiods.csv")

# Figure: MAE by year and quintile returns.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
yrs = by_year.index
for col, lab, c in [("analyst", "Analysts", SERIES[0]), ("ml_noA", "ML (no analyst input)", SERIES[1]),
                    ("ml_A", "ML + analysts", SERIES[2])]:
    axes[0].plot(yrs, 100 * by_year[col], marker="o", color=c, label=lab)
axes[0].set_title("Forecast error, 12-month EPS")
axes[0].set_ylabel("MAE (% of price)")
axes[0].legend(loc="upper left")
q = p_ew[[1, 2, 3, 4, 5]].mean()
axes[1].bar(range(1, 6), 100 * q.values, color=SERIES[0], width=0.6)
axes[1].set_xticks(range(1, 6))
axes[1].set_xlabel("Quintile of conditional bias (analyst minus ML)")
axes[1].set_ylabel("Next-month return (%, EW)")
axes[1].set_title(f"Q5-Q1 = {100*res.loc['EW','HL']:.2f}%/month (t={res.loc['EW','t']:.1f})")
axes[1].axhline(0, color=INK2, lw=0.8)
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
