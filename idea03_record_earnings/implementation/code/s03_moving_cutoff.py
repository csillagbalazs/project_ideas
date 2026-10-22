"""Section 5.2 (main result): the moving-cutoff design.

Sample: reports of firms below their record before the report (gap > 0). A report sets a record when
SUE > gap. Every surprise bin (0.25 SD wide) gets its own intercept, so the jump at r = 0 compares
firms with (almost) the same surprise that land on either side of the record only because their
pre-report gap differed. Local-linear, triangular kernel, SEs two-way clustered by month and firm.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import INK2, SERIES, binned, cell, fmt, load, macros, plt, rd, savefig, write

s = load()
b = s[s.at_rec == 0].copy()

outs = [("ar0", "Return,\\\\ month 0 (\\%)"), ("car1_3", "Return,\\\\ $+1$ to $+3$ (\\%)"),
        ("car1_6", "Return,\\\\ $+1$ to $+6$ (\\%)"), ("abto0", "Abnormal\\\\ turnover (\\%)"),
        ("rev", "Analyst\\\\ revision (bp)"), ("next_sue", "Next-quarter\\\\ SUE"), ("fut4", "EPS growth,\\\\ 4 quarters")]
R1 = {y: rd(b, y, h=1.0) for y, _ in outs}
R2 = {y: rd(b, y, h=2.0) for y, _ in outs}
RQ = {y: rd(b, y, h=2.0, poly=2) for y, _ in outs}


def shortlab(lab):
    return "\\shortstack{" + lab + "}"


head = " & " + " & ".join(shortlab(l) for _, l in outs) + "\\\\\n\\midrule\n"
lines = []
for lab, R in [("Jump at the record, $h=1$", R1), ("Jump at the record, $h=2$", R2),
               ("Local quadratic, $h=2$", RQ)]:
    c = [cell(R[y]["jump"], R[y]["t"]) for y, _ in outs]
    lines += [lab + " & " + " & ".join(x[0] for x in c) + r"\\", " & " + " & ".join(x[1] for x in c) + r"\\[1pt]"]
lines.append(r"\midrule")
lines.append("Mean just below the record & " + " & ".join(fmt(R1[y]["mean_below"]) for y, _ in outs) + r"\\")
lines.append("Reports ($h=1$) & " + " & ".join(f"{R1[y]['n']:,}" for y, _ in outs) + r"\\")
write("tab_main.tex", "\\begin{tabular}{l" + "c" * len(outs) + "}\n\\toprule\n" + head + "\n".join(lines)
      + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Balance: predetermined variables ------------------------------------------------------------
covs = [("size", "Log size"), ("bm", "Log B/M"), ("mom", "Momentum (\\%)"), ("pth", "Price/52w high"),
        ("car_pre", "Return, $-2$ to $-1$ (\\%)"), ("prev_sue", "Previous SUE"), ("covered", "Analyst coverage"),
        ("annual", "Annual report")]
BL = {y: rd(b, y, h=1.0) for y, _ in covs}
BL2 = {y: rd(b, y, h=2.0) for y, _ in covs}
lb = []
for y, lab in covs:
    c1, c2 = cell(BL[y]["jump"], BL[y]["t"], 3), cell(BL2[y]["jump"], BL2[y]["t"], 3)
    lb.append(f"{lab} & {c1[0]} & {c1[1]} & {c2[0]} & {c2[1]}\\\\")
write("tab_balance.tex", "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{$h=1$} & \\multicolumn{2}{c}{$h=2$}\\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Jump & $t$ & Jump & $t$\\\\\n\\midrule\n" + "\n".join(lb)
      + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Benchmark: how large is the earnings response? ---------------------------------------------
z = s[s.sue_rw.abs() <= 3].dropna(subset=["ar0", "car1_3"])
erc = {}
for y in ["ar0", "car1_3"]:
    zz = z[[y, "sue_rw", "ann_mi", "permno"]].copy()
    for c in [y, "sue_rw"]:
        zz[c] = zz[c] - zz.groupby("ann_mi")[c].transform("mean")
    r = sm.OLS(100 * zz[y], zz[["sue_rw"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(zz.ann_mi)[0]})
    erc[y] = r.params.iloc[0]
ub = {y: R1[y]["jump"] + 1.96 * R1[y]["se"] for y in ["ar0", "car1_3"]}
lb_ = {y: R1[y]["jump"] - 1.96 * R1[y]["se"] for y in ["ar0", "car1_3"]}
ub2 = {y: R2[y]["jump"] + 1.96 * R2[y]["se"] for y in ["ar0", "car1_3"]}
lb2 = {y: R2[y]["jump"] - 1.96 * R2[y]["se"] for y in ["ar0", "car1_3"]}

# ---- Figure 4: binned outcomes around the record --------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(7.7, 2.2))
for ax, (y, title, sc) in zip(axes, [("ar0", "(a) Return, month 0 (%)", 100), ("car1_3", "(b) Return, +1 to +3 (%)", 100),
                                       ("abto0", "(c) Abnormal turnover (%)", 1), ("next_sue", "(d) Next-quarter SUE", 1)]):
    m, se, d = binned(b, y, "r", width=0.2, h=2.0)
    ax.errorbar(m.index, sc * m.values, yerr=1.96 * sc * se.values, fmt="o", ms=3, color=SERIES[0], lw=0.8, capsize=0)
    for side in (-1, 1):
        q = d[np.sign(d.r) == side]
        p = sm.WLS(sc * q[y], sm.add_constant(q.r), weights=1 - q.r.abs() / 2).fit().params
        xs = np.linspace(0, 2 * side, 10)
        ax.plot(xs, p.iloc[0] + p.iloc[1] * xs, color=SERIES[1], lw=1.6)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_title(title, fontsize=8.2)
    ax.set_xlabel("Distance to record, r")
fig.tight_layout()
savefig(fig, "fig_main")

# ---- Figure 5: bandwidth sensitivity --------------------------------------------------------------
hs = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4), sharex=True)
bw_rows = []
for ax, (y, title) in zip(axes, [("ar0", "(a) Month 0"), ("car1_3", "(b) Months +1 to +3")]):
    est = [rd(b, y, h=h) for h in hs]
    for h, o in zip(hs, est):
        bw_rows.append({"y": y, "h": h, "jump": o["jump"], "se": o["se"], "n": o["n"]})
    ax.errorbar(hs, [o["jump"] for o in est], yerr=[1.96 * o["se"] for o in est], fmt="o", color=SERIES[0], ms=4,
                capsize=2.5, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Bandwidth h")
axes[0].set_ylabel("Jump at the record (%)")
fig.tight_layout()
savefig(fig, "fig_bandwidth")
pd.DataFrame(bw_rows).to_csv("../output/data/bandwidth.csv", index=False)

macros("main", {
    "MainN": f"{R1['ar0']['n']:,}", "MainNtwo": f"{R2['ar0']['n']:,}", "MainNbelow": f"{len(b):,}",
    "MainJumpA": fmt(R1["ar0"]["jump"]), "MainJumpAT": fmt(R1["ar0"]["t"]),
    "MainJumpD": fmt(R1["car1_3"]["jump"]), "MainJumpDT": fmt(R1["car1_3"]["t"]),
    "MainJumpS": fmt(R1["car1_6"]["jump"]), "MainJumpST": fmt(R1["car1_6"]["t"]),
    "MainJumpTO": fmt(R1["abto0"]["jump"]), "MainJumpTOT": fmt(R1["abto0"]["t"]),
    "MainJumpRev": fmt(R1["rev"]["jump"]), "MainJumpRevT": fmt(R1["rev"]["t"]),
    "MainJumpNext": fmt(R1["next_sue"]["jump"]), "MainJumpNextT": fmt(R1["next_sue"]["t"]),
    "MainJumpFut": fmt(R1["fut4"]["jump"]), "MainJumpFutT": fmt(R1["fut4"]["t"]),
    "MainJumpAtwo": fmt(R2["ar0"]["jump"]), "MainJumpAtwoT": fmt(R2["ar0"]["t"]),
    "MainJumpDtwo": fmt(R2["car1_3"]["jump"]), "MainJumpDtwoT": fmt(R2["car1_3"]["t"]),
    "CIAlo": fmt(lb_["ar0"]), "CIAhi": fmt(ub["ar0"]), "CIDlo": fmt(lb_["car1_3"]), "CIDhi": fmt(ub["car1_3"]),
    "CIAtwolo": fmt(lb2["ar0"]), "CIAtwohi": fmt(ub2["ar0"]), "CIDtwolo": fmt(lb2["car1_3"]), "CIDtwohi": fmt(ub2["car1_3"]),
    "ERCA": fmt(erc["ar0"]), "ERCD": fmt(erc["car1_3"]),
    "UBinSDA": fmt(ub2["ar0"] / erc["ar0"], 1),
    "BalSize": fmt(BL["size"]["jump"], 3), "BalSizeT": fmt(BL["size"]["t"]),
    "BalPrev": fmt(BL["prev_sue"]["jump"], 3), "BalPrevT": fmt(BL["prev_sue"]["t"]),
    "BalPrevTwo": fmt(BL2["prev_sue"]["jump"], 3), "BalPrevTwoT": fmt(BL2["prev_sue"]["t"]),
    "BalMaxT": fmt(max(abs(BL[y]["t"]) for y, _ in covs)),
    "BalMaxTtwo": fmt(max(abs(BL2[y]["t"]) for y, _ in covs)),
})
for y, _ in outs:
    print(f"{y:8s} h1 {R1[y]['jump']:7.2f} ({R1[y]['t']:5.2f}) n={R1[y]['n']} | h2 {R2[y]['jump']:7.2f} ({R2[y]['t']:5.2f}) | q2 {RQ[y]['jump']:7.2f} ({RQ[y]['t']:5.2f})")
for y, _ in covs:
    print(f"bal {y:8s} h1 {BL[y]['jump']:7.3f} ({BL[y]['t']:5.2f}) | h2 {BL2[y]['jump']:7.3f} ({BL2[y]['t']:5.2f})")
print("ERC", erc, "CI h1", lb_, ub, "CI h2", lb2, ub2)
