"""Section 5.2: the naive comparison and the information-set benchmark.

Naive test: the annual-report-month return responds to the annual surprise; does the slope
fall when the Q1 report arrives in the same month?
Benchmark: when the reports come in different months, the annual surprise is priced before
Q1 is known, so its slope also contains the market's forecast of Q1. Once Q1 is public,
part of that response is undone. Comparing like with like means comparing returns over a
window in which both reports are public for everyone.

Outputs: tab_naive_benchmark.tex, fig_naive_vs_aligned, macros_bench.tex
"""
import numpy as np
import pandas as pd

from utils import CLASS_COLOR, CLASS_ORDER, DATA, INK2, fmt, macros, plt, reg, regtable, savefig, write

s = pd.read_parquet(DATA / "sample.parquet")
s["SD"], s["SM"] = s.same_day, s.same_month
s["J"] = s.SD + s.SM
for k in ["4", "1"]:
    for c in ["SD", "SM", "J"]:
        s[f"s{k}_{c}"] = s[f"s{k}"] * s[c]
CL2 = ["ann_mi4", "firm"]

m1 = reg(s, "ar_m4", ["s4", "s4_SM", "s4_SD", "s1_J", "SM", "SD"], fes=["ann_mi4"], cluster=CL2)
sep, joint = s[s.separate == 1], s[s.J == 1]
m2 = reg(sep, "ar_m4", ["s4"], fes=["ann_mi4"], cluster=CL2)
m3 = reg(sep, "ar_m1", ["s1", "s4"], fes=["ann_mi1"], cluster=["ann_mi1", "firm"])
m4 = reg(joint, "ar_m4", ["s4", "s1"], fes=["ann_mi4"], cluster=CL2)
m5 = reg(s, "car_win", ["s4", "s1", "s4_SM", "s4_SD", "s1_SM", "s1_SD", "SM", "SD"], fes=["ann_mi1"],
         cluster=["ann_mi1", "firm"])
rows = [("s4", "$SUE^{A}$ (annual surprise)"), ("s4_SM", "$SUE^{A}\\times$ Same month"),
        ("s4_SD", "$SUE^{A}\\times$ Same day"), ("s1", "$SUE^{Q}$ (Q1 surprise)"),
        ("s1_J", "$SUE^{Q}\\times$ Same month or day"), ("s1_SM", "$SUE^{Q}\\times$ Same month"),
        ("s1_SD", "$SUE^{Q}\\times$ Same day")]
body = regtable([m1, m2, m3, m4, m5], rows, scale=100, digits=2,
                extra=[("Sample", ["All", "Separate", "Separate", "Same month/day", "All"]),
                       ("Return window", ["Annual month", "Annual month", "Q1 month", "Joint month", "Both public"])],
                header=["(1)", "(2)", "(3)", "(4)", "(5)"])
write("tab_naive_benchmark.tex", "\\begin{tabular}{lccccc}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}")
for i, m in enumerate([m1, m2, m3, m4, m5], 1):
    print(i, {v: (round(100 * m.params[v], 3), round(m.tvalues[v], 2)) for v in m.params.index}, m.nobs_used)

# Rational benchmark: separate firms' response to SUE^A once Q1 is out = (2) + SUE^A slope in (3).
rho = reg(sep, "s1", ["s4"], fes=["fyear"], cluster=["firm"])
pred_path = m2.params.s4 + m3.params.s4
pred_rho = m2.params.s4 - m3.params.s1 * rho.params.s4
print(f"rho={rho.params.s4:.3f}; predicted joint slope via path {100*pred_path:.3f}, via rho {100*pred_rho:.3f}; observed {100*m4.params.s4:.3f}")


def combo(res, terms):
    r = np.zeros(len(res.params))
    for t_ in terms:
        r[list(res.params.index).index(t_)] = 1
    tt = res.t_test(r)
    return 100 * float(np.squeeze(tt.effect)), 100 * float(np.squeeze(tt.sd))


naive = {"Separate months": combo(m1, ["s4"]), "Same month": combo(m1, ["s4", "s4_SM"]),
         "Same day": combo(m1, ["s4", "s4_SD"])}
aligned = {"Separate months": combo(m5, ["s4"]), "Same month": combo(m5, ["s4", "s4_SM"]),
           "Same day": combo(m5, ["s4", "s4_SD"])}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, d, title in [(axes[0], naive, "(a) Naive: annual-report month"),
                     (axes[1], aligned, "(b) Aligned: both reports public")]:
    for i, c in enumerate(CLASS_ORDER):
        b, se = d[c]
        ax.bar(i, b, color=CLASS_COLOR[c], width=0.6)
        ax.errorbar(i, b, yerr=1.96 * se, color=INK2, capsize=3, lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Separate\nmonths", "Same\nmonth", "Same\nday"])
    ax.set_title(title)
    ax.axhline(0, color=INK2, lw=0.8)
axes[0].set_ylabel("Return response to a 1-unit\nannual surprise (%)")
fig.tight_layout()
savefig(fig, "fig_naive_vs_aligned")

macros("bench", {
    "NaiveSep": fmt(naive["Separate months"][0]), "NaiveSD": fmt(naive["Same day"][0]),
    "NaiveSM": fmt(naive["Same month"][0]),
    "NaiveSDdiffT": fmt(m1.tvalues["s4_SD"]), "NaiveSMdiffT": fmt(m1.tvalues["s4_SM"]),
    "NaiveDropPct": f"{100 * (1 - naive['Same day'][0] / naive['Separate months'][0]):.0f}",
    "BenchCorrection": fmt(100 * m3.params.s4), "BenchCorrectionT": fmt(m3.tvalues.s4),
    "BenchPredPath": fmt(100 * pred_path), "BenchPredRho": fmt(100 * pred_rho),
    "BenchRho": fmt(rho.params.s4), "BenchQoneSep": fmt(100 * m3.params.s1),
    "BenchJoint": fmt(100 * m4.params.s4),
    "AlignedSep": fmt(aligned["Separate months"][0]), "AlignedSD": fmt(aligned["Same day"][0]),
    "AlignedSDdiff": fmt(100 * m5.params.s4_SD), "AlignedSDdiffT": fmt(m5.tvalues.s4_SD),
    "AlignedSMdiff": fmt(100 * m5.params.s4_SM), "AlignedSMdiffT": fmt(m5.tvalues.s4_SM),
    "AlignedQoneSDT": fmt(m5.tvalues.s1_SD),
})
print("done benchmark")
