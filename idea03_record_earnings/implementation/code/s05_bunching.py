"""Section 5.4: do firms manage earnings to set records? Bunching tests at the record and benchmarks.

Each test fits a Poisson regression of bin counts (0.05 SD bins) on a polynomial in the running
variable on each side and reports the excess density just above the threshold, exp(jump) - 1.
"""
import numpy as np
import pandas as pd

from utils import DATA, INK2, SERIES, cell, density, fmt, load, macros, plt, savefig, write

s = load()
allr = pd.read_parquet(DATA / "reports_all.parquet")
far = s[s.gap >= 1.0]

tests = [
    ("Record, all reports", s.r, "r"),
    ("Record, firms with gap $\\geq1$ (window $\\pm0.5$)", far.r, "far"),
    ("Beat last year (SUE $=0$), firms at their record", s[s.at_rec == 1].sue, "sue_at"),
    ("Beat last year (SUE $=0$), firms below, previous SUE $>0$", s[(s.at_rec == 0) & (s.prev_sue > 0)].sue, "sue_below"),
    ("Zero earnings (TTM EPS $=0$), all reports", allr.z_ttm, "zero"),
    ("Zero annual earnings, annual reports", allr[allr.fqtr == 4].z_ttm, "zero_fy"),
]
res = {}
lines = [" & \\multicolumn{2}{c}{Linear, $\\pm1$} & \\multicolumn{2}{c}{Quadratic, $\\pm1$} & "
         "\\multicolumn{2}{c}{Linear, $\\pm2$}\\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         " & Excess (\\%) & Reports & Excess (\\%) & Reports & Excess (\\%) & Reports\\\\", r"\midrule"]
for lab, x, key in tests:
    x = x.to_numpy()
    if key == "far":
        o = [density(x, h=0.5, bw=0.025), density(x, h=0.5, bw=0.025, poly=2), density(x, h=0.5, bw=0.05)]
    else:
        o = [density(x, h=1.0), density(x, h=1.0, poly=2), density(x, h=2.0)]
    res[key] = o
    cells = []
    for q in o:
        c = cell(q["jump"], q["t"], 1)
        cells += [f"{c[0]} {c[1]}", f"{q['n']:,}"]
    lines.append(lab + " & " + " & ".join(cells) + r"\\")
    if key == "far":
        lines[-1] = lines[-1].replace("Linear, $\\pm2$", "")
write("tab_bunching.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# Placebo cutoffs for the record test (all reports): how often does a smooth density give a 'jump'?
plac = {c: density(s.r.to_numpy(), h=1.0, cutoff=c) for c in (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)}

# ---- Figure 7 -------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.7, 2.4))
for ax, (key, title, xlab) in zip(axes, [("r", "(a) At the record", "Distance to record, r"),
                                         ("sue_at", "(b) SUE = 0, firms at record", "Surprise, SUE"),
                                         ("zero", "(c) At zero earnings", "TTM EPS / SD")]):
    o = res[key][0]
    d = o["bins"]
    ax.bar(d.x, d.c, width=0.045, color=SERIES[0], alpha=0.85)
    for side in (0, 1):
        q = d[d.D == side]
        ax.plot(q.x, o["fit"][q.index], color=SERIES[1], lw=1.6)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_title(title, fontsize=8.6)
    ax.set_xlabel(xlab)
    ax.text(0.97, 0.95, f"excess: {o['jump']:+.0f}%\n(t = {o['t']:.1f})", transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5)
axes[0].set_ylabel("Reports per 0.05 SD bin")
fig.tight_layout()
savefig(fig, "fig_bunching")

R = {k: v[0] for k, v in res.items()}
macros("bunch", {
    "BunchRec": fmt(R["r"]["jump"], 1), "BunchRecT": fmt(R["r"]["t"]),
    "BunchFar": fmt(R["far"]["jump"], 1), "BunchFarT": fmt(R["far"]["t"]),
    "BunchSueAt": fmt(R["sue_at"]["jump"], 0), "BunchSueAtT": fmt(R["sue_at"]["t"]),
    "BunchSueBelow": fmt(R["sue_below"]["jump"], 0), "BunchSueBelowT": fmt(R["sue_below"]["t"]),
    "BunchZero": fmt(R["zero"]["jump"], 0), "BunchZeroT": fmt(R["zero"]["t"]),
    "BunchZeroFY": fmt(R["zero_fy"]["jump"], 0), "BunchZeroFYT": fmt(R["zero_fy"]["t"]),
    "BunchPlacMaxT": fmt(max(abs(o["t"]) for o in plac.values())),
})
for k, v in res.items():
    print(k, [(round(q["jump"], 1), round(q["t"], 2), q["n"]) for q in v])
print("placebo", {c: (round(o["jump"], 1), round(o["t"], 2)) for c, o in plac.items()})
