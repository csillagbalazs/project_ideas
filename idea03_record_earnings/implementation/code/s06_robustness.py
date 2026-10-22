"""Section 5.5: robustness, heterogeneity, other record definitions, a benchmark threshold, placebos.

Every row re-estimates the jump at the record of the main design (Table 4, h = 1) for three outcomes.
"""
import numpy as np
import pandas as pd

from utils import DATA, INK2, SERIES, fmt, load, macros, neg, plt, rd, savefig, stars, write

s = load()
b = s[s.at_rec == 0].copy()
b["sbin_year"] = b.sbin.astype(int).astype(str) + "_" + b.year.astype(str)
b["small"] = (b["size"] < b.groupby("year")["size"].transform("median")).astype(float)
allr = pd.read_parquet(DATA / "reports_all.parquet")
Y = ["ar0", "car1_3", "abto0"]


def alt(run, gapcol, cond=None):
    d = allr[np.isfinite(allr[run]) & (allr[run] != 0) & (allr[gapcol].abs() > 1e-9)].copy()
    if cond is not None:
        d = d[cond(d)]
    return d.rename(columns={run: "r"})


ctrl = ("prev_sue", "size_f", "mom_f", "pth_f")
rows = [
    ("Specification", None),
    ("Baseline ($h=1$, 0.25 SD surprise bins, triangular kernel)", dict(d=b)),
    ("Bandwidth $h=0.5$", dict(d=b, h=0.5)),
    ("Bandwidth $h=2$", dict(d=b, h=2.0)),
    ("Local quadratic, $h=2$", dict(d=b, h=2.0, poly=2)),
    ("Uniform kernel", dict(d=b, kernel="uni")),
    ("Surprise bins of 0.1 SD", dict(d=b, fe="sbin10")),
    ("Surprise bins of 0.5 SD", dict(d=b, fe="sbin50")),
    ("Surprise bin $\\times$ year effects", dict(d=b, fe="sbin_year")),
    ("XGBoost surprise bins", dict(d=b, fe="xbin")),
    ("Donut: drop $|r|<0.05$", dict(d=b, donut=0.05)),
    ("Controls: previous SUE, size, momentum, price/52w high", dict(d=b, extra=ctrl)),
    ("Subsamples", None),
    ("Small firms", dict(d=b[b.small == 1])),
    ("Large firms", dict(d=b[b.small == 0])),
    ("2005--2012", dict(d=b[b.year <= 2012])),
    ("2013--2020", dict(d=b[b.year >= 2013])),
    ("Annual reports", dict(d=b[b.annual == 1])),
    ("Interim reports", dict(d=b[b.annual == 0])),
    ("Record set 8 or more quarters ago", dict(d=b[b.q_since >= 8])),
    ("Record set less than 8 quarters ago", dict(d=b[b.q_since < 8])),
    ("No analyst coverage", dict(d=b[b.covered == 0])),
    ("Analyst coverage", dict(d=b[b.covered == 1])),
    ("Other record definitions", None),
    ("Single-quarter EPS record", dict(d=alt("r_q", "gap_q"))),
    ("Year-to-date EPS record (same period of earlier years)", dict(d=alt("r_ytd", "gap_ytd"))),
    ("Annual EPS record (annual reports)", dict(d=alt("r_fy", "gap_fy"))),
    ("Benchmark threshold", None),
    ("Zero earnings (TTM EPS $=0$)", dict(d=alt("z_ttm", "z_ttm"))),
    ("Placebo cutoffs (one side of the record only, $h=0.5$)", None),
    ("Baseline at the true record, $h=0.5$", dict(d=b, h=0.5)),
    ("Placebo at $r=-1$", dict(d=b[b.r < 0], h=0.5, cutoff=-1.0)),
    ("Placebo at $r=-0.5$", dict(d=b[b.r < 0], h=0.5, cutoff=-0.5)),
    ("Placebo at $r=+0.5$", dict(d=b[b.r > 0], h=0.5, cutoff=0.5)),
    ("Placebo at $r=+1$", dict(d=b[b.r > 0], h=0.5, cutoff=1.0)),
]
out, lines = [], []
for lab, spec in rows:
    if spec is None:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}}\\\\")
        continue
    kw = dict(spec)
    d = kw.pop("d")
    cells = []
    for y in Y:
        o = rd(d, y, **kw)
        out.append({"row": lab, "y": y, "jump": o["jump"], "se": o["se"], "t": o["t"], "n": o["n"]})
        cells.append(neg(f"{o['jump']:.2f}") + stars(o["t"]) + " " + neg(f"({o['t']:.2f})"))
    lines.append("\\quad " + lab + " & " + " & ".join(cells) + r"\\")
write("tab_robustness.tex", "\\begin{tabular}{lccc}\n\\toprule\n & Return, month 0 & Return, $+1$ to $+3$ & "
      "Abnormal turnover\\\\\n & (\\%) & (\\%) & (\\%)\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
O = pd.DataFrame(out)
O.to_csv(DATA / "robustness.csv", index=False)

# ---- Figure 8: forest plot ------------------------------------------------------------------------
labs = [lab for lab, spec in rows if spec is not None]
short = {"Baseline ($h=1$, 0.25 SD surprise bins, triangular kernel)": "Baseline",
         "Controls: previous SUE, size, momentum, price/52w high": "With controls",
         "Year-to-date EPS record (same period of earlier years)": "Year-to-date record",
         "Baseline at the true record, $h=0.5$": "True record, h = 0.5"}
fig, axes = plt.subplots(1, 3, figsize=(7.7, 5.6), sharey=True)
ypos = np.arange(len(labs))[::-1]
for ax, y, title in zip(axes, Y, ["Return, month 0 (%)", "Return, +1 to +3 (%)", "Abnormal turnover (%)"]):
    q = O[O.y == y].set_index("row").loc[labs]
    grey = [("Placebo" in l) or ("Zero earnings" in l) for l in labs]
    for yy, (_, r_), gr in zip(ypos, q.iterrows(), grey):
        ax.errorbar(r_.jump, yy, xerr=1.96 * r_.se, fmt="o", ms=3.5, color=INK2 if gr else SERIES[0], lw=1, capsize=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title, fontsize=8.4)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([short.get(l, l).replace("$\\times$", "x").replace("$", "").replace("\\geq", ">=")
                         .replace("|r|<", "|r| < ").replace("--", "-") for l in labs], fontsize=7)
fig.tight_layout()
savefig(fig, "fig_forest")

main = O[O.row.str.startswith("Baseline (")]
sig = O[~O.row.str.contains("Placebo|Zero|true record")]
zero = O[O.row.str.contains("Zero earnings")].set_index("y")
macros("rob", {
    "RobNrows": f"{sig.row.nunique()}", "RobNtests": f"{len(sig)}",
    "RobNsig": f"{int((sig.t.abs() >= 1.96).sum())}",
    "RobNsigExp": f"{0.05 * len(sig):.1f}",
    "ZeroTO": fmt(zero.loc["abto0", "jump"]), "ZeroTOT": fmt(zero.loc["abto0", "t"]),
    "ZeroA": fmt(zero.loc["ar0", "jump"]), "ZeroAT": fmt(zero.loc["ar0", "t"]),
})
print(O.assign(jump=O.jump.round(2), t=O.t.round(2))[["row", "y", "jump", "t", "n"]].to_string(index=False))
