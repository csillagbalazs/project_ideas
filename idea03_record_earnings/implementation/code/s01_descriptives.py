"""Descriptive statistics (Table 2) and Figures 1-2: a worked example, record frequency, the gap."""
import numpy as np
import pandas as pd

from utils import INK, INK2, SERIES, fmt, load, macros, plt, savefig, write

s = load()

# ---- Table 2: means by record status ------------------------------------------------------------
groups = {
    "All": s,
    "Extended": s[(s.rec == 1) & (s.at_rec == 1)],
    "Regained": s[(s.rec == 1) & (s.at_rec == 0)],
    "Near miss": s[(s.r < 0) & (s.r >= -1)],
    "Far below": s[s.r < -1],
}
rows = [
    ("Reports", lambda d: f"{len(d):,}"),
    ("Share of reports (\\%)", lambda d: f"{100 * len(d) / len(s):.1f}"),
    ("Surprise, SUE", lambda d: fmt(d.sue_rw.mean())),
    ("Gap to record before report (median)", lambda d: fmt(d.gap.median())),
    ("Quarters since record (median)", lambda d: f"{d.q_since.median():.0f}"),
    ("Log market value", lambda d: fmt(d["size"].mean())),
    ("Log book-to-market", lambda d: fmt(d.bm.mean())),
    ("Momentum, months $-12$ to $-2$ (\\%)", lambda d: fmt(100 * d.mom.mean(), 1)),
    ("Price / 52-week high", lambda d: fmt(d.pth.mean())),
    ("Analyst coverage (\\%)", lambda d: f"{100 * d.covered.mean():.0f}"),
    ("Annual report (\\%)", lambda d: f"{100 * d.annual.mean():.0f}"),
    ("Return, announcement month (\\%)", lambda d: fmt(100 * d.ar0.mean())),
    ("Return, months $+1$ to $+3$ (\\%)", lambda d: fmt(100 * d.car1_3.mean())),
    ("Abnormal turnover (\\%)", lambda d: fmt(d.abto0.mean(), 1)),
    ("Next-quarter SUE", lambda d: fmt(d.next_sue.mean())),
]
body = [" & " + " & ".join(groups) + r"\\", r"\midrule"]
for lab, f in rows:
    body.append(lab + " & " + " & ".join(f(d) for d in groups.values()) + r"\\")
write("tab_descriptives.tex", "\\begin{tabular}{lccccc}\n\\toprule\n & & \\multicolumn{2}{c}{New record} & "
      "\\multicolumn{2}{c}{No record}\\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n" + "\n".join(body)
      + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Worked example: Wuliangye -------------------------------------------------------------------
w = s[s.name.str.contains("WULIANGYE")].sort_values(["fyear", "fqtr"]).copy()
ex = w[(w.fyear == 2018) & (w.fqtr == 1)].iloc[0]
# ---- Figure 1: (a) Wuliangye's TTM EPS and its record; (b) same surprise, different gap ---------
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
w["lab"] = w.fyear + (w.fqtr - 1) / 4
w["record_line"] = w.ttm.cummax()
ax.plot(w.lab, w.ttm, color=SERIES[0], lw=1.8, label="TTM EPS")
ax.step(w.lab, w.record_line, where="post", color=INK2, lw=1, ls="--", label="Best-ever TTM EPS (record)")
rec = w[w.rec == 1]
ax.scatter(rec.lab, rec.ttm, s=14, color=SERIES[1], zorder=3, label="New record")
xe = ex.fyear + (ex.fqtr - 1) / 4
ax.annotate(f"Q1 2018: record regained\nafter {ex.q_since:.0f} quarters", xy=(xe, ex.ttm), xytext=(xe - 5.3, ex.ttm + 0.75),
            fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.annotate("2013-15: anti-graft campaign\nhits liquor sales", xy=(2014.6, 1.55), xytext=(2013.2, 0.35), fontsize=7.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.set_title("(a) Wuliangye: trailing EPS and its record")
ax.set_ylabel("EPS (yuan, split-adjusted)")
ax.legend(loc="upper left", fontsize=7)
ax.set_xlim(2005.5, 2020.8)

ax = axes[1]
for i, (gap, lab) in enumerate([(0.8, "Firm A"), (1.2, "Firm B")]):
    c = SERIES[1] if gap < 1 else SERIES[0]
    end = -gap + 1.0
    ax.annotate("", xy=(i, end), xytext=(i, -gap), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax.scatter([i], [-gap], color=c, s=22, zorder=3)
    ax.text(i - 0.07, (end - gap) / 2, "SUE\n= 1.0", fontsize=7.5, color=INK, va="center", ha="right")
    ax.text(i + 0.07, -gap, f"gap = {gap}", fontsize=7.5, color=INK2, va="center")
    ax.text(i + 0.07, end, "new record" if end > 0 else "just short", fontsize=7.5, va="center", color=INK,
            fontweight="bold")
ax.axhline(0, color=INK, lw=1)
ax.text(-0.45, 0.04, "record", fontsize=7.5, color=INK, ha="left", va="bottom")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Firm A", "Firm B"])
ax.set_xlim(-0.5, 1.75)
ax.set_ylim(-1.5, 0.6)
ax.set_ylabel("TTM EPS minus record (SD units)")
ax.set_title("(b) Same news, different side")
fig.tight_layout()
savefig(fig, "fig_example")

# ---- Figure 2: (a) record frequency by year; (b) the gap; (c) the (SUE, gap) plane ---------------
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
ax = axes[0]
yr = s.groupby("year").agg(ext=("rec", lambda v: 0), n=("rec", "size"))
yr["ext"] = s[(s.rec == 1) & (s.at_rec == 1)].groupby("year").size() / yr.n * 100
yr["reg"] = s[(s.rec == 1) & (s.at_rec == 0)].groupby("year").size() / yr.n * 100
ax.bar(yr.index, yr.ext, color=SERIES[1], width=0.8, label="Extended (gap = 0)")
ax.bar(yr.index, yr.reg, bottom=yr.ext, color=SERIES[0], width=0.8, label="Regained (gap > 0)")
ax.set_title("(a) Reports setting a record, %")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylim(0, 45)
ax = axes[1]
pos = s[s.gap > 0].gap.clip(upper=6)
share0 = 100 * (s.gap == 0).mean()
cnt, edges = np.histogram(pos, bins=np.arange(0, 6.01, 0.25))
ax.bar(edges[:-1] + 0.125, 100 * cnt / len(s), width=0.22, color=SERIES[0])
ax.bar([-0.2], [share0], width=0.22, color=SERIES[1])
ax.text(0.05, share0 - 1.5, f"gap = 0: {share0:.0f}% of reports\n(the last report was the record)", fontsize=7, color=INK, va="top")
ax.set_title("(b) Gap before the report")
ax.set_xlabel("Gap (SD units; last bin 6+)")
ax.set_ylabel("% of reports")
ax = axes[2]
rng = np.random.default_rng(1)
z = s[(s.gap <= 4) & (s.sue.abs() <= 4)]
z = z.iloc[rng.choice(len(z), 5000, replace=False)]
jit = np.where(z.gap == 0, rng.uniform(-0.04, 0.04, len(z)), 0)
for v, c, lab in [(0, SERIES[0], "No record"), (1, SERIES[1], "New record")]:
    q = z[z.rec == v]
    ax.scatter(q.sue, q.gap + jit[z.rec.to_numpy() == v], s=2, alpha=0.35, color=c, label=lab, rasterized=True)
ax.plot([0, 4], [0, 4], color=INK, lw=1.1)
ax.text(2.45, 1.25, "record line:\nSUE = gap", fontsize=7, color=INK)
ax.set_xlabel("Surprise, SUE")
ax.set_ylabel("Gap before report")
ax.set_title("(c) Who sets a record")
ax.legend(fontsize=7, loc="upper left", markerscale=4, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
fig.tight_layout()
savefig(fig, "fig_records")

macros("desc", {
    "ExName": "Wuliangye", "ExQuarters": f"{ex.q_since:.0f}", "ExTTMprev": f"{ex.ttm_prev:.2f}",
    "ExRecord": f"{ex.prevmax:.2f}", "ExTTM": f"{ex.ttm:.2f}", "ExEPS": f"{ex.eps_actual:.2f}",
    "ExEPSlag": f"{ex.eps_lag4:.2f}", "ExSUE": f"{ex.sue:.1f}", "ExGap": f"{ex.gap:.1f}", "ExR": f"{ex.r:.1f}",
    "ExSD": f"{ex.sd_ue:.3f}",
    "RecShareEarly": f"{100 * s[s.year <= 2012].rec.mean():.0f}", "RecShareLate": f"{100 * s[s.year > 2012].rec.mean():.0f}",
    "MeanSUEext": fmt(groups["Extended"].sue_rw.mean()), "MeanSUEreg": fmt(groups["Regained"].sue_rw.mean()),
    "QsinceReg": f"{groups['Regained'].q_since.median():.0f}",
    "ShareRegained": f"{100 * len(groups['Regained']) / len(s):.1f}",
})
print("example", ex[["name", "fyear", "fqtr", "eps_actual", "eps_lag4", "ttm_prev", "prevmax", "ttm", "sue", "gap", "r"]].to_dict())
print(yr.round(1).to_string())
