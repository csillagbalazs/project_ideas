"""Section 5.1: the naive RD at the record (the design of the proof of concept) and why it misleads.

Naive design: local-linear RD in r with a jump and a change in slope (kink) at r = 0, controlling
linearly for good and bad surprises (SUE+ and SUE-), as in the proof of concept.
"""
import numpy as np
import statsmodels.api as sm

from utils import INK2, SERIES, cell, fmt, load, macros, plt, rd, savefig, write

s = load()
X = ("sue_pos", "sue_neg")
below = s[s.at_rec == 0]

# ---- Panel A: returns ------------------------------------------------------------------------------
specs = [("ar0", s, 1.0), ("ar0", s, 2.0), ("car1_3", s, 1.0), ("car1_3", s, 2.0), ("car1_3", below, 1.0),
         ("car1_3", below, 2.0)]
A = [rd(d, y, h=h, fe=None, extra=X) for y, d, h in specs]
head = (" & \\multicolumn{2}{c}{Announcement month} & \\multicolumn{2}{c}{Months $+1$ to $+3$} & "
        "\\multicolumn{2}{c}{Months $+1$ to $+3$}\\\\\n & \\multicolumn{4}{c}{All reports} & "
        "\\multicolumn{2}{c}{Gap $>0$ only}\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\n"
        " & $h=1$ & $h=2$ & $h=1$ & $h=2$ & $h=1$ & $h=2$\\\\\n\\midrule\n")
lines = []
for key, lab in [("jump", "Jump at the record"), ("kink", "Kink: change in slope")]:
    c = [cell(o[key], o["t" if key == "jump" else "kt"]) for o in A]
    lines += [lab + " & " + " & ".join(x[0] for x in c) + r"\\", " & " + " & ".join(x[1] for x in c) + r"\\[1pt]"]
lines.append("Reports & " + " & ".join(f"{o['n']:,}" for o in A) + r"\\")
panelA = head + "\n".join(lines)

# ---- Panel B: the same design applied to variables fixed before the report -----------------------
covs = [("size", "Log size"), ("mom", "Momentum (\\%)"), ("pth", "Price/52w high"), ("prev_sue", "Previous SUE"),
        ("car_pre", "Pre-report return (\\%)")]
B = [rd(s, y, h=1.0, fe=None, extra=X) for y, _ in covs]
headB = (" & " + " & ".join(l for _, l in covs) + "\\\\\n\\midrule\n")
lb = []
for key, lab in [("jump", "Jump at the record"), ("kink", "Kink: change in slope")]:
    c = [cell(o[key], o["t" if key == "jump" else "kt"]) for o in B]
    lb += [lab + " & " + " & ".join(x[0] for x in c) + r"\\", " & " + " & ".join(x[1] for x in c) + r"\\[1pt]"]
panelB = headB + "\n".join(lb)
write("tab_naive.tex",
      "\\begin{tabular}{lcccccc}\n\\toprule\n\\multicolumn{7}{l}{\\textit{A. Size-adjusted returns (\\%)}}\\\\\n"
      + panelA + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{6pt}\n"
      "\\begin{tabular}{lccccc}\n\\toprule\n\\multicolumn{6}{l}{\\textit{B. Placebo outcomes fixed before the report "
      "(all reports, $h=1$)}}\\\\\n" + panelB + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Figure 3 -------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.55))


def panel(ax, d, y, title, ylab, net=True, scale=100):
    z = d[d.r.abs() <= 2].dropna(subset=[y, *X]).copy()
    if net:
        res = sm.OLS(z[y], sm.add_constant(z[list(X)])).fit()
        z["v"] = res.resid + z[y].mean()
    else:
        z["v"] = z[y]
    z["b"] = (np.floor(z.r / 0.2) + 0.5) * 0.2
    bm = z.groupby("b").v.mean()
    ax.scatter(bm.index, scale * bm.values, s=10, color=SERIES[0], zorder=3)
    for side in (-1, 1):
        q = z[np.sign(z.r) == side]
        b = sm.OLS(scale * q.v, sm.add_constant(q.r)).fit().params
        xs = np.linspace(0, 2 * side, 10)
        ax.plot(xs, b.iloc[0] + b.iloc[1] * xs, color=SERIES[1], lw=1.6)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_title(title)
    ax.set_xlabel("Distance to record, r (SD units)")
    ax.set_ylabel(ylab)


panel(axes[0], s, "car1_3", "(a) Drift, all reports (PoC)", "Return net of surprise (%)")
panel(axes[1], below, "car1_3", "(b) Drift, firms below record", "Return net of surprise (%)")
panel(axes[2], s, "size", "(c) Firm size, all reports", "Log size net of surprise", scale=1)
fig.tight_layout()
savefig(fig, "fig_naive")

o1, o2, ob = A[2], A[3], A[4]
macros("naive", {
    "NaiveJumpA": fmt(A[0]["jump"]), "NaiveJumpAT": fmt(A[0]["t"]),
    "NaiveKinkD": fmt(o1["kink"]), "NaiveKinkDT": fmt(o1["kt"]), "NaiveKinkDtwo": fmt(o2["kink"]), "NaiveKinkDtwoT": fmt(o2["kt"]),
    "NaiveJumpD": fmt(o1["jump"]), "NaiveJumpDT": fmt(o1["t"]),
    "NaiveKinkBelow": fmt(ob["kink"]), "NaiveKinkBelowT": fmt(ob["kt"]),
    "NaiveKinkBelowTwo": fmt(A[5]["kink"]), "NaiveKinkBelowTwoT": fmt(A[5]["kt"]),
    "NaiveKinkSize": fmt(B[0]["kink"]), "NaiveKinkSizeT": fmt(B[0]["kt"]),
    "NaiveKinkMom": fmt(B[1]["kink"]), "NaiveKinkMomT": fmt(B[1]["kt"]),
    "NaiveKinkPth": fmt(B[2]["kink"]), "NaiveKinkPthT": fmt(B[2]["kt"]),
    "NaiveKinkPrev": fmt(B[3]["kink"]), "NaiveKinkPrevT": fmt(B[3]["kt"]),
})
for (y, d, h), o in zip(specs, A):
    print(f"{y} h={h} n={o['n']}: jump {o['jump']:.2f} ({o['t']:.2f}) kink {o['kink']:.2f} ({o['kt']:.2f})")
for (y, _), o in zip(covs, B):
    print(f"cov {y}: jump {o['jump']:.3f} ({o['t']:.2f}) kink {o['kink']:.3f} ({o['kt']:.2f})")
