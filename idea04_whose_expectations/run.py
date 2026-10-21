"""Idea 4 PoC: whose earnings expectations do Chinese prices reflect?

The SRW surprise splits into a part the XGBoost forecast predicted before the
announcement (PRED = (f_xgb - f_rw)/sd) and the XGBoost surprise (sue_xgb):
    sue_rw = PRED + sue_xgb.
If prices embed ML-quality expectations, PRED should not move announcement-month
returns. If they embed naive SRW expectations, PRED and sue_xgb move them equally.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements, factors  # noqa: E402
from common.stats import alpha, fama_macbeth, nw_mean, sort_portfolios  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

a = announcements()
a = a[(a.year.between(2005, 2020)) & a.sue_rw.notna() & a.sue_xgb.notna() & a.ar0.notna()].copy()
a["pred"] = a.sue_rw - a.sue_xgb
a["covered"] = a.eps_12fwd.notna().astype(float)
for c in ["size", "bm", "mom", "turnover"]:
    a[c] = a.groupby("ann_mi")[c].transform(lambda s: s.fillna(s.median()))
ctrl = ["size", "bm", "mom", "turnover"]
print("N announcements:", len(a), "months:", a.ann_mi.nunique())

rows = {}
# 1) Horse race and decomposition, Fama-MacBeth by announcement month.
specs = {
    "(1) SRW": ["sue_rw"],
    "(2) XGB": ["sue_xgb"],
    "(3) Decomp.": ["pred", "sue_xgb"],
    "(4) + controls": ["pred", "sue_xgb", *ctrl],
}
tabs = {}
for dep in ["ar0", "car1_3", "car_pre"]:
    for name, xs in specs.items():
        r = fama_macbeth(a, dep, xs, t="ann_mi", min_obs=50)
        series = r.attrs["series"]
        # Average cross-sectional R2 is not returned; compute separately for (1)/(2)
        tabs[(dep, name)] = r
        print(dep, name, r.round(4).to_dict("index"))


def avg_r2(df, y, x):
    out = []
    for _, g in df[["ann_mi", y, x]].dropna().groupby("ann_mi"):
        if len(g) >= 50:
            out.append(np.corrcoef(g[y], g[x])[0, 1] ** 2)
    return np.mean(out)


r2 = {x: avg_r2(a, "ar0", x) for x in ["sue_rw", "sue_xgb"]}
print("avg monthly R2 ar0:", r2)

# Equality test: coef(pred) == coef(sue_xgb) on the FM series.
ser = tabs[("ar0", "(3) Decomp.")].attrs["series"]
diff_m, diff_t = nw_mean(ser.pred - ser.sue_xgb)
ratio = ser.pred.mean() / ser.sue_xgb.mean()
print(f"pred - xgb coef diff {diff_m:.4f} t={diff_t:.2f}; ratio {ratio:.2f}")

# 2) Heterogeneity: small vs large, covered vs not.
het = {}
a["small"] = (a.groupby("ann_mi")["size"].rank(pct=True) <= 0.5).astype(float)
for lab, sub in {"Small": a[a.small == 1], "Large": a[a.small == 0],
                 "No analyst": a[a.covered == 0], "Analyst": a[a.covered == 1]}.items():
    r = fama_macbeth(sub, "ar0", ["pred", "sue_xgb"], t="ann_mi", min_obs=30)
    het[lab] = r
    print(lab, r.round(4).to_dict("index"))

# 3) Tradable strategy: sort announcers on PRED at the start of the announcement month.
#    Use only forecasts whose inputs were all public before the month began.
src = pd.read_parquet(Path(__file__).resolve().parents[2] / "earnings_prediction/output/forecasts/eps_forecasts_oos.parquet",
                      columns=["permno", "fyear", "fqtr", "max_input_date"])
t = a.merge(src, on=["permno", "fyear", "fqtr"], how="left")
t = t[t.max_input_date < t.ann_date.dt.to_period("M").dt.to_timestamp()]
t = t.drop_duplicates(["permno", "ann_mi"], keep="last")
p = sort_portfolios(t, "pred", "ar0", n=5, t="ann_mi")
fac = factors()
raw = t.groupby("ann_mi").r0.mean()  # not used for alpha; HL is self-financing
hl_m, hl_t = nw_mean(p.HL)
a3, t3, n3 = alpha(p.HL, fac, "CH3")
a4, t4, n4 = alpha(p.HL, fac, "CH4")
print(f"Strategy N={len(t)} months={len(p)}: HL abn {hl_m:.4f} (t={hl_t:.2f}); CH3 a={a3:.4f} t={t3:.2f}; CH4 a={a4:.4f} t={t4:.2f}")
pm = p.mean()

# ---------------- outputs ----------------
names = {"sue_rw": "SUE (SRW)", "sue_xgb": "SUE (XGBoost)", "pred": "Predictable part (PRED)",
         "size": "Size", "bm": "log B/M", "mom": "Momentum", "turnover": "Turnover", "const": "Constant"}
order = ["sue_rw", "sue_xgb", "pred", "size", "bm", "mom", "turnover"]
lines = []
for v in order:
    cells, tcells = [], []
    for name in specs:
        r = tabs[("ar0", name)]
        if v in r.index:
            cells.append(f"{100*r.loc[v,'coef']:.2f}"); tcells.append(f"({r.loc[v,'t']:.1f})")
        else:
            cells.append(""); tcells.append("")
    lines.append(f"{names[v]} & " + " & ".join(cells) + r"\\")
    lines.append(" & " + " & ".join(tcells) + r"\\")
drift = tabs[("car1_3", "(3) Decomp.")]
pre = tabs[("car_pre", "(3) Decomp.")]
lines.append(r"\midrule")
lines.append(r"Avg.\ monthly $R^2$ & " + f"{100*r2['sue_rw']:.2f}\\% & {100*r2['sue_xgb']:.2f}\\% & & " + r"\\")
(OUT / "table_fm.tex").write_text("\n".join(lines))

summary = pd.DataFrame({
    "stat": ["N", "months", "coef_pred_ar0", "t_pred_ar0", "coef_xgb_ar0", "t_xgb_ar0", "ratio", "diff_t",
             "coef_pred_drift", "t_pred_drift", "coef_xgb_drift", "t_xgb_drift",
             "coef_pred_pre", "t_pred_pre", "coef_xgb_pre", "t_xgb_pre",
             "r2_rw", "r2_xgb", "hl", "hl_t", "a3", "t3", "a4", "t4", "n_strat", "months_strat"],
    "value": [len(a), a.ann_mi.nunique(),
              tabs[("ar0", "(3) Decomp.")].loc["pred", "coef"], tabs[("ar0", "(3) Decomp.")].loc["pred", "t"],
              tabs[("ar0", "(3) Decomp.")].loc["sue_xgb", "coef"], tabs[("ar0", "(3) Decomp.")].loc["sue_xgb", "t"],
              ratio, diff_t,
              drift.loc["pred", "coef"], drift.loc["pred", "t"], drift.loc["sue_xgb", "coef"], drift.loc["sue_xgb", "t"],
              pre.loc["pred", "coef"], pre.loc["pred", "t"], pre.loc["sue_xgb", "coef"], pre.loc["sue_xgb", "t"],
              r2["sue_rw"], r2["sue_xgb"], hl_m, hl_t, a3, t3, a4, t4, len(t), len(p)]})
het_df = pd.concat({k: v[["coef", "t"]] for k, v in het.items()}).unstack(0)
summary.to_csv(OUT / "summary_stats.csv", index=False)
het_df.to_csv(OUT / "heterogeneity.csv")
p.to_csv(OUT / "strategy_quintiles.csv")

# Figure: binned returns by decile of each component (announcement month and drift).
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
for ax, dep, title in [(axes[0], "ar0", "Announcement month: mostly the ML surprise"),
                       (axes[1], "car1_3", "Months +1 to +3: both parts drift")]:
    for col, lab, c in [("pred", "Predictable part (PRED)", SERIES[0]), ("sue_xgb", "XGBoost surprise", SERIES[1])]:
        q = np.ceil(10 * a.groupby("ann_mi")[col].rank(pct=True)).clip(1, 10)
        b = a.groupby(q)[dep].mean()
        ax.plot(b.index, 100 * b.values, marker="o", color=c, label=lab)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decile within announcement month")
    ax.set_title(title)
    ax.axhline(0, color=INK2, lw=0.8)
axes[0].set_ylabel("Size-adjusted return (%)")
axes[0].legend(loc="upper left")
fig.tight_layout()
save(fig, OUT / "fig_main.pdf")
print(het_df.round(4))
print("quintiles", p.mean().round(4).to_dict())
