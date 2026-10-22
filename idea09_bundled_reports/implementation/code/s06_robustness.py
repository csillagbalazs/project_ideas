"""Section 5.5: heterogeneity, robustness, burying bad news, and a placebo.

Every row re-estimates the main specification (2) of s04: release-month size-adjusted return on
Bundle x SUE^A and Bundle x SUE^Q with date-specific intercepts and slopes, two-way clustered.
Outputs: tab_robustness.tex, fig_forest, macros_rob.tex
"""
import numpy as np
import pandas as pd

from utils import DATA, INK2, SERIES, fmt, macros, plt, reg_slopes, savefig, write

s = pd.read_parquet(DATA / "sample.parquet")
s = s.sort_values(["permno", "fyear"])
nxt = s.groupby("permno").same_day.shift(-1)
s["SD_next"] = nxt.where(s.groupby("permno").fyear.diff(-1) == -1)
t = s[s.both_same_month == 1].copy()
t["SD"] = t.same_day.astype(float)
t["small"] = (t.groupby("fyear")["size"].rank(pct=True) <= 0.5).astype(int)
CL = ["d4", "firm"]


def run(df, a="s4", q="s1", treat="SD", dep="ar_m4"):
    df = df.copy()
    df["A_T"], df["Q_T"], df["T"] = df[a] * df[treat], df[q] * df[treat], df[treat]
    r = reg_slopes(df.rename(columns={a: "A", q: "Q"}), dep, ["T", "A_T", "Q_T"], [("d4", ["A", "Q"])], CL)
    return r


tests = [
    ("Baseline", t, {}),
    ("Small firms", t[t.small == 1], {}),
    ("Large firms", t[t.small == 0], {}),
    ("No analyst coverage", t[t.covered == 0], {}),
    ("Analyst coverage", t[t.covered == 1], {}),
    ("Reporting years 2006--2012", t[t.fyear <= 2011], {}),
    ("Reporting years 2013--2020", t[t.fyear >= 2012], {}),
    ("Excluding 2020 (COVID delays)", t[t.fyear <= 2018], {}),
    ("Excluding annual losses", t[t.loss == 0], {}),
    ("XGBoost annual surprise", t, {"a": "x4"}),
    ("Release month + next month", t, {"dep": "car_m4_next"}),
    ("Placebo: bundles next year", t[t.SD == 0].dropna(subset=["SD_next"]), {"treat": "SD_next"}),
]
rows, res = [], {}
for name, df, kw in tests:
    r = run(df, **kw)
    res[name] = r
    rows.append({"test": name, "bA": 100 * r.params.A_T, "seA": 100 * r.bse.A_T, "tA": r.tvalues.A_T,
                 "bQ": 100 * r.params.Q_T, "seQ": 100 * r.bse.Q_T, "tQ": r.tvalues.Q_T, "n": r.nobs_used})
R = pd.DataFrame(rows)
print(R.round(3).to_string(index=False))

# Burying bad news: does a bundle soften the response to *bad* annual news?
b = t.copy()
b["neg"] = b.neg4.astype(float)
b["A_neg"] = b.s4 * b.neg
for c in ["s4", "s1", "A_neg", "neg"]:
    b[f"{c}_T"] = b[c] * b.SD
bury = reg_slopes(b, "ar_m4", ["SD", "s4_T", "A_neg_T", "s1_T", "neg_T"], [("d4", ["s4", "s1", "neg", "A_neg"])], CL)
b["janjul"] = b.car_janapr + b.car_mayjul
bury2 = reg_slopes(b, "janjul", ["SD", "s4_T", "A_neg_T", "s1_T", "neg_T"], [("d4", ["s4", "s1", "neg", "A_neg"])], CL)
print("bury", {v: (round(100 * bury.params[v], 3), round(bury.tvalues[v], 2)) for v in bury.params.index})
print("bury Jan-Jul", {v: (round(100 * bury2.params[v], 3), round(bury2.tvalues[v], 2)) for v in bury2.params.index})

def cell(b_, t_):
    neg = lambda x: x.replace("-", "\\ensuremath{-}")
    st = "***" if abs(t_) >= 2.576 else "**" if abs(t_) >= 1.96 else "*" if abs(t_) >= 1.645 else ""
    return neg(f"{b_:.2f}") + st + " " + neg(f"({t_:.2f})")

lines = [r"\begin{tabular}{lccr}", r"\toprule",
         r" & $SUE^{A}\times$ Bundle & $SUE^{Q}\times$ Bundle & $N$\\", r"\midrule"]
for _, r in R.iterrows():
    lines.append(f"{r.test} & {cell(r.bA, r.tA)} & {cell(r.bQ, r.tQ)} & {int(r.n):,}\\\\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{4}{l}{\emph{Burying bad annual news?} Extra response to bad annual news in a bundle:}\\")
lines.append(f"\\quad Release month & {cell(100 * bury.params.A_neg_T, bury.tvalues.A_neg_T)} & & {bury.nobs_used:,}\\\\")
lines.append(f"\\quad January--July & {cell(100 * bury2.params.A_neg_T, bury2.tvalues.A_neg_T)} & & {bury2.nobs_used:,}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_robustness.tex", "\n".join(lines))

fig, axes = plt.subplots(1, 2, figsize=(3.7, 3.4), sharey=True)
n = len(R)
for ax, b_, se, title in [(axes[0], "bA", "seA", "$SUE^{A}\\times$Bundle"), (axes[1], "bQ", "seQ", "$SUE^{Q}\\times$Bundle")]:
    for i, r in R.iterrows():
        col = INK2 if r.test.startswith("Placebo") else SERIES[0]
        ax.errorbar(r[b_], -i, xerr=1.96 * r[se], fmt="o", color=col, ms=3.5, capsize=2, lw=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title, fontsize=8)
axes[0].set_yticks([-i for i in range(n)])
axes[0].set_yticklabels([x.replace("--", "–") for x in R.test], fontsize=6.5)
fig.supxlabel("Difference in response (%), 95% CI", fontsize=7.5)
fig.tight_layout()
savefig(fig, "fig_forest")

sig = int(((R.tA.abs() >= 1.96) | (R.tQ.abs() >= 1.96)).sum())
macros("rob", {
    "RobNtests": f"{len(R)}", "RobNsig": f"{sig}",
    "RobSmallA": fmt(R.set_index('test').loc['Small firms', 'bA']), "RobSmallAT": fmt(R.set_index('test').loc['Small firms', 'tA']),
    "RobPlaceboQ": fmt(R.set_index('test').loc['Placebo: bundles next year', 'bQ']),
    "RobPlaceboQT": fmt(R.set_index('test').loc['Placebo: bundles next year', 'tQ']),
    "BuryRel": fmt(100 * bury.params.A_neg_T), "BuryRelT": fmt(bury.tvalues.A_neg_T),
    "BuryJJ": fmt(100 * bury2.params.A_neg_T), "BuryJJT": fmt(bury2.tvalues.A_neg_T),
})
print("done robustness")
