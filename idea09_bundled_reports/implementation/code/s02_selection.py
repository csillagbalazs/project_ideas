"""Section 5.1: who bundles? Linear probability models of same-day release.

Outputs: tab_selection.tex, fig_selection, macros_sel.tex
"""
import numpy as np
import pandas as pd

from utils import DATA, INK2, SERIES, demean, fmt, macros, plt, reg, regtable, savefig, write

s = pd.read_parquet(DATA / "sample.parquet")
s["SD"] = s.same_day.astype(float)
s["late10"] = 10 * s.late_rank
s["size_z"] = s.groupby("fyear")["size"].transform(lambda x: (x - x.mean()) / x.std())
s["size_z"] = s.size_z.fillna(0)
X_all = ["late10", "neg4", "neg1", "loss", "size_z", "covered"]
X_date = ["neg4", "neg1", "loss", "size_z", "covered"]

m1 = reg(s, "SD", X_all, fes=["fyear"], cluster=["firm"])
m2 = reg(s, "SD", X_all, fes=["fyear", "firm"], cluster=["firm"])
m3 = reg(s, "SD", X_date, fes=["d4"], cluster=["firm", "d4"])
m4 = reg(s, "SD", X_date, fes=["d4", "firm"], cluster=["firm", "d4"])
rows = [("late10", "Lateness of annual report (per decile)"), ("neg4", "Bad annual news ($SUE^{A}<0$)"),
        ("neg1", "Bad Q1 news ($SUE^{Q}<0$)"), ("loss", "Annual loss"), ("size_z", "Size (standardised)"),
        ("covered", "Analyst coverage")]
body = regtable([m1, m2, m3, m4], rows, scale=100, digits=2,
                extra=[("Year fixed effects", ["Yes", "Yes", "--", "--"]),
                       ("Annual-report-date fixed effects", ["--", "--", "Yes", "Yes"]),
                       ("Firm fixed effects", ["--", "Yes", "--", "Yes"])],
                header=["(1)", "(2)", "(3)", "(4)"])
write("tab_selection.tex", "\\begin{tabular}{lcccc}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}")
for i, m in enumerate([m1, m2, m3, m4], 1):
    print(i, {v: (round(100 * m.params[v], 2), round(m.tvalues[v], 2)) for v in m.params.index})

# Figure: P(same day) by surprise decile, after removing annual-report-date effects.
d = s.dropna(subset=["SD", "s4", "s1"]).copy()
d["SD_res"] = demean(d, ["SD"], ["d4"])["SD"] + d.SD.mean()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for col, lab, c in [("s4", "Annual surprise", SERIES[2]), ("s1", "Q1 surprise", SERIES[0])]:
    q = pd.qcut(d[col].rank(method="first"), 10, labels=False) + 1
    b = d.groupby(q).SD_res.mean()
    ax.plot(b.index, 100 * b.values, marker="o", ms=3.5, color=c, label=lab)
ax.set_xticks(range(1, 11))
ax.set_xlabel("Surprise decile (1 = worst news)")
ax.set_ylabel("% same-day release\n(same annual-report date)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
savefig(fig, "fig_selection")

macros("sel", {
    "SelLate": fmt(100 * m1.params["late10"], 1),
    "SelLateT": fmt(m1.tvalues["late10"], 1),
    "SelNegQdate": fmt(100 * m3.params["neg1"], 1),
    "SelNegQdateT": fmt(m3.tvalues["neg1"], 1),
    "SelNegQfirm": fmt(100 * m4.params["neg1"], 1),
    "SelNegQfirmT": fmt(m4.tvalues["neg1"], 1),
    "SelNegAdate": fmt(100 * m3.params["neg4"], 1),
    "SelNegAdateT": fmt(m3.tvalues["neg4"], 1),
    "SelNegAyear": fmt(100 * m1.params["neg4"], 1),
    "SelNegAyearT": fmt(m1.tvalues["neg4"], 1),
    "SelBaseRate": f"{100 * s.SD.mean():.0f}",
})
print("done selection")
