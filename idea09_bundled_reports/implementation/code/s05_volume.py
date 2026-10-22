"""Section 5.4: trading volume. Attention shows up in trading even when prices are hard to read.

Abnormal turnover in the release month = log turnover minus the firm's mean over months -12..-2.
If a bundle diverts attention, turnover should respond less to the size of the bundled surprises.
Sample and designs as in s04 (both reports in the same calendar month).
Outputs: tab_volume.tex, fig_volume, macros_vol.tex
"""
import numpy as np
import pandas as pd

from utils import CLASS_COLOR, DATA, demean, fmt, macros, plt, reg, reg_slopes, regtable, savefig, write

s = pd.read_parquet(DATA / "sample.parquet")
t = s[s.both_same_month == 1].copy()
t["SD"] = t.same_day.astype(float)
t["a4"], t["a1"] = t.s4.abs(), t.s1.abs()
t["a4_SD"], t["a1_SD"] = t.a4 * t.SD, t.a1 * t.SD
X = ["SD", "a4_SD", "a1_SD"]
CL = ["d4", "firm"]
m1 = reg(t, "abto_m4", ["a4", "a1", *X], fes=["d4"], cluster=CL)
m2 = reg_slopes(t, "abto_m4", X, [("d4", ["a4", "a1"])], CL)
m3 = reg(t, "abto_m4", ["a4", "a1", *X], fes=["d4", "firm"], cluster=CL)
m4 = reg_slopes(t, "abto_m4", X, [("firm", ["a4", "a1"]), ("d4", ["a4", "a1"])], CL)
rows = [("a4", "$|SUE^{A}|$"), ("a1", "$|SUE^{Q}|$"), ("a4_SD", "$|SUE^{A}|\\times$ Bundle"),
        ("a1_SD", "$|SUE^{Q}|\\times$ Bundle"), ("SD", "Bundle")]
body = regtable([m1, m2, m3, m4], rows, scale=100, digits=2,
                extra=[("Date-specific intercepts", ["Yes"] * 4),
                       ("Date-specific slopes", ["--", "Yes", "--", "Yes"]),
                       ("Firm fixed effects / slopes", ["--", "--", "FE", "Slopes"])],
                header=["(1)", "(2)", "(3)", "(4)"])
write("tab_volume.tex", "\\begin{tabular}{lcccc}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}")
for i, m in enumerate([m1, m2, m3, m4], 1):
    print(i, {v: (round(100 * m.params[v], 3), round(m.tvalues[v], 2)) for v in m.params.index}, m.nobs_used)

d = t.dropna(subset=["abto_m4", "a1", "a4"]).copy()
d["y"] = demean(d, ["abto_m4"], ["d4"])["abto_m4"] + d.abto_m4.mean()
fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.6), sharey=True)
for ax, col, title in [(axes[0], "a4", "$|SUE^{A}|$"), (axes[1], "a1", "$|SUE^{Q}|$")]:
    d["q"] = pd.qcut(d[col].rank(method="first"), 5, labels=False) + 1
    for sd, name in [(0, "Same month"), (1, "Same day")]:
        b = d[d.SD == sd].groupby("q").y.mean()
        ax.plot(b.index, 100 * b.values, marker="o", ms=3.5, color=CLASS_COLOR[name],
                label=name if sd == 0 else "Same day (bundle)")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel(f"{title} quintile", fontsize=7.5)
axes[0].set_ylabel("Abnormal turnover in\nrelease month (%)")
axes[1].legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
savefig(fig, "fig_volume")

macros("vol", {
    "VolAtwo": fmt(100 * m2.params.a4_SD), "VolAtwoT": fmt(m2.tvalues.a4_SD),
    "VolQtwo": fmt(100 * m2.params.a1_SD), "VolQtwoT": fmt(m2.tvalues.a1_SD),
    "VolQfour": fmt(100 * m4.params.a1_SD), "VolQfourT": fmt(m4.tvalues.a1_SD),
    "VolBaseQ": fmt(100 * m1.params.a1), "VolBaseA": fmt(100 * m1.params.a4),
})
print("done volume")
