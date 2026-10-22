"""Section 3: sample construction, descriptive statistics, and the rise of bundling.

Outputs: tab_sample_flow.tex, tab_descriptives.tex, fig_timeline, fig_bundling_trends, macros_desc.tex
"""
import numpy as np
import pandas as pd

from utils import CLASS_COLOR, CLASS_ORDER, DATA, INK2, SERIES, fmt, macros, plt, savefig, write

s = pd.read_parquet(DATA / "sample.parquet")
flow = pd.read_csv(DATA / "sample_flow.csv")

# ------------------------------------------------------------------ Table 1: sample flow
lines = [r"\begin{tabular}{lr}", r"\toprule", r"Step & Firm-years\\", r"\midrule"]
for _, r in flow.iterrows():
    lines.append(f"{r.step} & {int(r.n):,}\\\\")
lines.append(r"\midrule")
for c in CLASS_ORDER[::-1]:
    lines.append(f"\\quad {c} & {int((s.cls == c).sum()):,}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_sample_flow.tex", "\n".join(lines))

# ------------------------------------------------------------------ Table 2: descriptives by class
vars_ = [
    ("lag4", "Annual report: days after fiscal year end", 0),
    ("gap", "Days between annual and Q1 report", 1),
    ("april4", "Annual report released in April", 2),
    ("s4", "Annual surprise $SUE^{A}$", 2),
    ("s1", "Q1 surprise $SUE^{Q}$", 2),
    ("neg4", "Bad annual news ($SUE^{A}<0$)", 2),
    ("neg1", "Bad Q1 news ($SUE^{Q}<0$)", 2),
    ("loss", "Annual loss", 2),
    ("size", "Size (log market value)", 2),
    ("covered", "Analyst coverage", 2),
    ("ar_m4", "Return, annual-report month (\\%)", 2),
    ("abto_m4", "Abnormal turnover, annual-report month", 2),
]
g = s.groupby("cls")
lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r" & Separate months & Same month & Same day\\", r"\midrule",
         "Firm-years & " + " & ".join(f"{(s.cls == c).sum():,}" for c in CLASS_ORDER) + r"\\"]
for v, lab, d in vars_:
    vals = []
    for c in CLASS_ORDER:
        x = s.loc[s.cls == c, v]
        x = 100 * x if v == "ar_m4" else x
        vals.append(fmt(x.mean(), d))
    lines.append(f"{lab} & " + " & ".join(vals) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_descriptives.tex", "\n".join(lines))

# ------------------------------------------------------------------ Figure 1: timeline schematic
fig, ax = plt.subplots(figsize=(7.0, 2.3))
day = lambda m, d: (pd.Timestamp(2019, m, d) - pd.Timestamp(2019, 1, 1)).days
rows = [("Firm A: separate months", [("Annual", day(3, 20)), ("Q1", day(4, 25))], CLASS_COLOR["Separate months"]),
        ("Firm B: same month", [("Annual", day(4, 12)), ("Q1", day(4, 26))], CLASS_COLOR["Same month"]),
        ("Firm C: same day (bundle)", [("Annual + Q1", day(4, 26))], CLASS_COLOR["Same day"])]
for i, (lab, evs, col) in enumerate(rows):
    y = 2 - i
    ax.hlines(y, 0, day(5, 31), color="#e4e3df", lw=1)
    for txt, x in evs:
        ax.plot(x, y, "o", color=col, ms=8, zorder=3)
        ax.annotate(txt, (x, y), xytext=(-4 if "+" in txt else 0, 7), textcoords="offset points",
                    ha="right" if "+" in txt else "center", fontsize=7.5, color=INK2)
    ax.text(-3, y, lab, ha="right", va="center", fontsize=8)
ax.axvline(day(4, 30), color=INK2, lw=1, ls="--")
ax.text(day(4, 30) + 2, 2.45, "30 April: deadline for\nboth reports", fontsize=7, color=INK2, va="top")
ax.axvspan(day(4, 1), day(4, 30), color="#f2f1ee", zorder=0)
ax.set_xticks([day(m, 1) for m in range(1, 6)] + [day(5, 31)])
ax.set_xticklabels(["1 Jan", "1 Feb", "1 Mar", "1 Apr", "1 May", "31 May"])
ax.set_yticks([])
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(-2, day(5, 31))
ax.grid(False)
for sp in ["left"]:
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.22)
savefig(fig, "fig_timeline")

# ------------------------------------------------------------------ Figure 2: rise of bundling
shares = s.groupby("fyear").cls.value_counts(normalize=True).unstack().fillna(0)
lag = s.groupby("fyear").lag4.median()
s["late_bin"] = np.ceil(20 * s.late_rank).clip(1, 20)
by_late = s.groupby("late_bin").same_day.mean()
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
for c in CLASS_ORDER:
    axes[0].plot(shares.index + 1, 100 * shares[c], marker="o", ms=3, color=CLASS_COLOR[c], label=c)
    yl = 100 * shares[c].iloc[-1]
    axes[0].annotate(c.replace(" months", "").replace("Separate", "Separate"), (shares.index[-1] + 1, yl),
                     xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color=CLASS_COLOR[c])
axes[0].set_title("(a) Release pattern")
axes[0].set_ylabel("% of firm-years")
axes[0].set_xlabel("Reporting year")
axes[0].set_xlim(2005.5, 2024.5)
axes[1].plot(lag.index + 1, lag.values, marker="o", ms=3, color=SERIES[0])
axes[1].set_title("(b) Annual-report lag")
axes[1].set_ylabel("Median days after year end")
axes[1].set_xlabel("Reporting year")
axes[2].bar(by_late.index, 100 * by_late.values, color=SERIES[0], width=0.8)
axes[2].set_title("(c) Bundling by lateness")
axes[2].set_xlabel("Lateness of annual report\n(within-year percentile, 5% bins)")
axes[2].set_ylabel("% same-day releases")
axes[2].set_xticks([1, 5, 10, 15, 20])
axes[2].set_xticklabels(["5", "25", "50", "75", "100"])
for ax in axes[:2]:
    ax.set_xticks([2006, 2012, 2018])
fig.tight_layout()
savefig(fig, "fig_bundling_trends")

# ------------------------------------------------------------------ macros
pre = s[s.fyear <= 2010].same_day.mean()
post = s[s.fyear == 2019].same_day.mean()
macros("desc", {
    "NFirmYears": f"{len(s):,}",
    "NFirms": f"{s.permno.nunique():,}",
    "NSameDay": f"{s.same_day.sum():,}",
    "NSameMonth": f"{s.same_month.sum():,}",
    "NSeparate": f"{s.separate.sum():,}",
    "ShareSDearly": f"{100 * pre:.0f}",
    "ShareSDlate": f"{100 * post:.0f}",
    "LagEarly": f"{s[s.fyear <= 2014].groupby('fyear').lag4.median().mean():.0f}",
    "LagLate": f"{lag.loc[2019]:.0f}",
    "ShareSDtopLate": f"{100 * s[s.late_rank > 0.8].same_day.mean():.0f}",
    "ShareSDbottomHalf": f"{100 * s[s.late_rank <= 0.5].same_day.mean():.0f}",
    "MedianDaySD": f"{s[s.same_day == 1].ann_date4.dt.day.median():.0f}",
    "MedianGapSM": f"{s[s.same_month == 1].gap.median():.0f}",
    "CorrSurprises": f"{s[['s4', 's1']].corr().iloc[0, 1]:.2f}",
})
print(shares.round(3).T.to_string())
print("done descriptives")
