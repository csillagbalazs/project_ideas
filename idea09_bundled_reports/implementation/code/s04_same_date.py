"""Section 5.3 (main test): bundlers versus firms that filed the annual report on the same day.

Sample: firm-years whose annual and Q1 reports fall in the same calendar month, so by the end
of that month every firm has published both reports. Treatment: Q1 report on the same day as
the annual report (bundle) versus a few days later (same month).

Specifications (two-way clustered by annual-report date and firm):
  (1) date fixed effects (intercepts) and common slopes,
  (2) date-specific slopes: every annual-report date gets its own response to SUE^A and SUE^Q,
  (3) firm-specific slopes (plus date effects),
  (4) firm- and date-specific slopes,
  (5) release month and the next month,
  (6) the three months after the release month (drift).
Outputs: tab_same_date.tex, fig_event_paths, fig_coefplot, macros_main.tex
"""
import numpy as np
import pandas as pd

from utils import (CLASS_COLOR, DATA, INK2, SERIES, fmt, macros, plt, reg, reg_slopes, regtable,
                   savefig, write)

s = pd.read_parquet(DATA / "sample.parquet")
t = s[s.both_same_month == 1].copy()
t["SD"] = t.same_day.astype(float)
for k in ["4", "1"]:
    t[f"s{k}_SD"] = t[f"s{k}"] * t.SD
X = ["SD", "s4_SD", "s1_SD"]
CL = ["d4", "firm"]
DS = [("d4", ["s4", "s1"])]

m1 = reg(t, "ar_m4", ["s4", "s1", *X], fes=["d4"], cluster=CL)
m2 = reg_slopes(t, "ar_m4", X, DS, CL)
m3 = reg_slopes(t, "ar_m4", X, [("firm", ["s4", "s1"]), ("d4", [])], CL)
m4 = reg_slopes(t, "ar_m4", X, [("firm", ["s4", "s1"]), ("d4", ["s4", "s1"])], CL)
m5 = reg_slopes(t, "car_m4_next", X, DS, CL)
m6 = reg_slopes(t, "car_post3", X, DS, CL)
models = [m1, m2, m3, m4, m5, m6]
rows = [("s4", "$SUE^{A}$"), ("s1", "$SUE^{Q}$"), ("s4_SD", "$SUE^{A}\\times$ Bundle"),
        ("s1_SD", "$SUE^{Q}\\times$ Bundle"), ("SD", "Bundle")]
body = regtable(models, rows, scale=100, digits=2,
                extra=[("Return window", ["Release", "Release", "Release", "Release", "Release+1", "Next 3"]),
                       ("Date-specific intercepts", ["Yes"] * 6),
                       ("Date-specific slopes", ["--", "Yes", "--", "Yes", "Yes", "Yes"]),
                       ("Firm-specific slopes", ["--", "--", "Yes", "Yes", "--", "--"])],
                header=[f"({i})" for i in range(1, 7)])
write("tab_same_date.tex", "\\begin{tabular}{lcccccc}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}")
for i, m in enumerate(models, 1):
    print(i, {v: (round(100 * m.params[v], 3), round(m.tvalues[v], 2)) for v in m.params.index}, m.nobs_used)

# How large an effect can we rule out? 95% bounds relative to the common slope in (1).
def bound(m, v):
    lo, hi = 100 * (m.params[v] - 1.96 * m.bse[v]), 100 * (m.params[v] + 1.96 * m.bse[v])
    return lo, hi
loA, hiA = bound(m2, "s4_SD")
loQ, hiQ = bound(m2, "s1_SD")
baseA, baseQ = 100 * m1.params.s4, 100 * m1.params.s1
print(f"95% CI SUE^A x Bundle [{loA:.2f},{hiA:.2f}] vs base {baseA:.2f}; SUE^Q x Bundle [{loQ:.2f},{hiQ:.2f}] vs base {baseQ:.2f}")

# ---------------------------------------------------------------- event-time cumulative paths
ks = list(range(-3, 4))
paths = {}
for j, k in enumerate(ks):
    cols = [f"r{i:+d}" for i in range(-3, k + 1)]
    t["cum"] = t[cols].sum(axis=1, min_count=len(cols))
    r = reg(t, "cum", ["s4", "s1", *X], fes=["d4"], cluster=CL)
    for sv in ["s4", "s1"]:
        base = r.params[sv]
        vec = np.zeros(len(r.params)); idx = list(r.params.index)
        vec[idx.index(sv)] = 1; vec[idx.index(f"{sv}_SD")] = 1
        tt = r.t_test(vec)
        paths[(sv, k)] = {"sm": 100 * base, "sm_se": 100 * r.bse[sv],
                          "sd": 100 * float(np.squeeze(tt.effect)), "sd_se": 100 * float(np.squeeze(tt.sd)),
                          "diff": 100 * r.params[f"{sv}_SD"], "diff_t": r.tvalues[f"{sv}_SD"]}
P = pd.DataFrame(paths).T
P.index.names = ["surprise", "month"]
P.to_csv(DATA / "event_paths.csv")
print(P.round(3))

lab = {-3: "Jan", -2: "Feb", -1: "Mar", 0: "Apr\n(release)", 1: "May", 2: "Jun", 3: "Jul"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, sv, title in [(axes[0], "s4", "(a) Annual surprise $SUE^{A}$"), (axes[1], "s1", "(b) Q1 surprise $SUE^{Q}$")]:
    q = P.loc[sv]
    for key, name in [("sm", "Same month"), ("sd", "Same day")]:
        c = CLASS_COLOR[name]
        ax.plot(ks, q[key], marker="o", ms=3.5, color=c, label=name if name != "Same day" else "Same day (bundle)")
        ax.fill_between(ks, q[key] - 1.96 * q[f"{key}_se"], q[key] + 1.96 * q[f"{key}_se"], color=c, alpha=0.12, lw=0)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(ks)
    ax.set_xticklabels([lab[k] for k in ks], fontsize=7)
    ax.set_title(title)
axes[0].set_ylabel("Cumulative return response\nsince January (%)")
axes[1].legend(fontsize=7, loc="upper left")
fig.tight_layout()
savefig(fig, "fig_event_paths")

# ---------------------------------------------------------------- coefficient plot across specs
fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.9), sharey=True)
names = ["(1) Date FE", "(2) Date slopes", "(3) Firm slopes", "(4) Firm+date slopes", "(5) Release+1", "(6) Next 3"]
for ax, v, title in [(axes[0], "s4_SD", "$SUE^{A}\\times$Bundle"), (axes[1], "s1_SD", "$SUE^{Q}\\times$Bundle")]:
    for i, m in enumerate(models):
        b, se = 100 * m.params[v], 100 * m.bse[v]
        ax.errorbar(b, -i, xerr=1.96 * se, fmt="o", color=SERIES[0], ms=4, capsize=2, lw=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title, fontsize=8)
axes[0].set_yticks([-i for i in range(len(models))])
axes[0].set_yticklabels(names, fontsize=7)
fig.supxlabel("Difference in response (%), 95% CI", fontsize=7.5)
fig.tight_layout()
savefig(fig, "fig_coefplot")

macros("main", {
    "MainN": f"{m1.nobs_used:,}", "MainNdates": f"{t.d4.nunique():,}",
    "MainNmixed": f"{(t.groupby('d4').SD.nunique() == 2).sum():,}",
    "MainNSD": f"{int(t.SD.sum()):,}", "MainNSM": f"{int((1 - t.SD).sum()):,}",
    "MainBaseA": fmt(baseA), "MainBaseQ": fmt(baseQ),
    "MainAoneD": fmt(100 * m1.params.s4_SD), "MainAoneT": fmt(m1.tvalues.s4_SD),
    "MainQoneD": fmt(100 * m1.params.s1_SD), "MainQoneT": fmt(m1.tvalues.s1_SD),
    "MainAtwoD": fmt(100 * m2.params.s4_SD), "MainAtwoT": fmt(m2.tvalues.s4_SD),
    "MainQtwoD": fmt(100 * m2.params.s1_SD), "MainQtwoT": fmt(m2.tvalues.s1_SD),
    "MainQfourD": fmt(100 * m4.params.s1_SD), "MainQfourT": fmt(m4.tvalues.s1_SD),
    "MainAfourD": fmt(100 * m4.params.s4_SD), "MainAfourT": fmt(m4.tvalues.s4_SD),
    "MainCIAlo": fmt(loA), "MainCIAhi": fmt(hiA), "MainCIQlo": fmt(loQ), "MainCIQhi": fmt(hiQ),
    "MainRuleOutQ": f"{100 * max(0.0, -loQ) / baseQ:.0f}", "MainRuleOutA": f"{100 * max(0.0, -loA) / baseA:.0f}",
    "MainDriftQ": fmt(100 * m6.params.s1_SD), "MainDriftQT": fmt(m6.tvalues.s1_SD),
    "MainDriftA": fmt(100 * m6.params.s4_SD), "MainDriftAT": fmt(m6.tvalues.s4_SD),
    "MainBundleLevelPost": fmt(100 * m6.params.SD), "MainBundleLevelPostT": fmt(m6.tvalues.SD),
})
print("done main")
