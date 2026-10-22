"""Section 5.3: firms already at their record (gap = 0). For them a record is the same event as SUE > 0.

Difference in discontinuities at SUE = 0: firms at their record (for whom SUE > 0 sets a record) versus
firms far below it (gap > h, for whom no surprise in the window sets a record). Month fixed effects;
every RD term is interacted with the at-record indicator; SEs two-way clustered by month and firm.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import INK2, SERIES, cell, diff_rd, fmt, load, macros, plt, savefig, write

s = load()


def sample(h, pos_prev=False):
    d = s[(s.at_rec == 1) | (s.gap > h)]
    if pos_prev:
        d = d[(d.at_rec == 1) | (d.prev_sue > 0)]
    return d


specs = [("ar0", 1.0, (), False, "Month 0"), ("car1_3", 1.0, (), False, "$+1$ to $+3$"),
         ("car1_3", 2.0, (), False, "$+1$ to $+3$"), ("car1_3", 1.0, (), True, "$+1$ to $+3$"),
         ("next_sue", 1.0, (), False, "Next SUE"), ("car1_3", 1.0, ("next_sue",), False, "$+1$ to $+3$")]
E = [diff_rd(sample(h, pp), y, "at_rec", h=h, extra=ex) for y, h, ex, pp, _ in specs]
rows = [("x", "Slope below SUE $=0$, far-below firms"), ("xG", "\\quad Extra slope, at-record firms"),
        ("D", "Jump at SUE $=0$, far-below firms"), ("DG", "\\quad Extra jump, at-record firms"),
        ("xD", "Kink at SUE $=0$, far-below firms"), ("xDG", "\\quad Extra kink, at-record firms")]
lines = [" & " + " & ".join(f"({i + 1})" for i in range(len(specs))) + r"\\",
         " & " + " & ".join(lab for *_, lab in specs) + r"\\", r"\midrule"]
for k, lab in rows:
    c = [cell(o[k], o[k + "_t"]) for o in E]
    lines += [lab + " & " + " & ".join(x[0] for x in c) + r"\\", " & " + " & ".join(x[1] for x in c) + r"\\[1pt]"]
lines.append(r"\midrule")
lines.append("Bandwidth $h$ & " + " & ".join(f"{h:g}" for _, h, *_ in specs) + r"\\")
lines.append("Far-below firms: previous SUE $>0$ & " + " & ".join("yes" if pp else "" for *_, pp, _ in specs) + r"\\")
lines.append("Controls for next-quarter SUE & " + " & ".join("yes" if ex else "" for _, _, ex, _, _ in specs) + r"\\")
lines.append("Reports & " + " & ".join(f"{o['n']:,}" for o in E) + r"\\")
lines.append("\\quad at-record firms & " + " & ".join(f"{o['nG']:,}" for o in E) + r"\\")
write("tab_at_record.tex", "\\begin{tabular}{l" + "c" * len(specs) + "}\n\\toprule\n" + "\n".join(lines)
      + "\n\\bottomrule\n\\end{tabular}\n")

# ---- Figure 6: response curves of at-record and far-below firms -----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.7, 2.45))
d = sample(1.5)
d = d[d.sue.abs() <= 1.5].copy()
for ax, (y, title, sc) in zip(axes, [("ar0", "(a) Return, month 0 (%)", 100), ("car1_3", "(b) Return, +1 to +3 (%)", 100),
                                       ("next_sue", "(c) Next-quarter SUE", 1)]):
    z = d.dropna(subset=[y]).copy()
    z["v"] = z[y] - z.groupby("ann_mi")[y].transform("mean") + z[y].mean()
    z["b"] = (np.floor(z.sue / 0.25) + 0.5) * 0.25
    for gval, col, lab in [(0, SERIES[0], "Far below record"), (1, SERIES[1], "At record (SUE > 0 = new record)")]:
        q = z[z.at_rec == gval]
        m = q.groupby("b").v.mean()
        ax.scatter(m.index, sc * m.values, s=11, color=col, zorder=3, label=lab)
        for side in (-1, 1):
            qq = q[np.sign(q.sue) == side]
            p = sm.WLS(sc * qq.v, sm.add_constant(qq.sue), weights=1 - qq.sue.abs() / 1.5).fit().params
            xs = np.linspace(0, 1.5 * side, 10)
            ax.plot(xs, p.iloc[0] + p.iloc[1] * xs, color=col, lw=1.5)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_title(title, fontsize=8.4)
    ax.set_xlabel("Surprise, SUE")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
savefig(fig, "fig_at_record")

o_car, o_next, o_ctrl = E[1], E[4], E[5]


def lincom(o, terms):
    """Sum of coefficients (scaled like the outcome) and its t-statistic."""
    res = o["res"]
    w = np.array([1.0 if k in terms else 0.0 for k in res.params.index])
    tt = res.t_test(w)
    sc = o["xG"] / res.params["xG"] if res.params["xG"] != 0 else 1.0
    return float(np.squeeze(tt.effect)) * sc, float(np.squeeze(tt.tvalue))


slopes = {}
for key, o in [("A", E[0]), ("D", o_car), ("N", o_next)]:
    slopes[key] = {"farbad": lincom(o, ["x"]), "fargood": lincom(o, ["x", "xD"]),
                   "atbad": lincom(o, ["x", "xG"]), "atgood": lincom(o, ["x", "xG", "xD", "xDG"])}
print({k: {kk: (round(v[0], 2), round(v[1], 2)) for kk, v in d.items()} for k, d in slopes.items()})
macros("atrec", {
    "AtKinkD": fmt(o_car["xDG"]), "AtKinkDT": fmt(o_car["xDG_t"]),
    "AtKinkDtwo": fmt(E[2]["xDG"]), "AtKinkDtwoT": fmt(E[2]["xDG_t"]),
    "AtKinkDpos": fmt(E[3]["xDG"]), "AtKinkDposT": fmt(E[3]["xDG_t"]),
    "AtKinkA": fmt(E[0]["xDG"]), "AtKinkAT": fmt(E[0]["xDG_t"]),
    "AtJumpA": fmt(E[0]["DG"]), "AtJumpAT": fmt(E[0]["DG_t"]),
    "AtJumpD": fmt(o_car["DG"]), "AtJumpDT": fmt(o_car["DG_t"]),
    "AtKinkNext": fmt(o_next["xDG"]), "AtKinkNextT": fmt(o_next["xDG_t"]),
    "AtKinkCtrl": fmt(o_ctrl["xDG"]), "AtKinkCtrlT": fmt(o_ctrl["xDG_t"]),
    "AtNextCoef": fmt(o_ctrl["next_sue"]), "AtNextCoefT": fmt(o_ctrl["next_sue_t"]),
    "AtKinkShare": f"{100 * (1 - o_ctrl['xDG'] / o_car['xDG']):.0f}",
    "AtNG": f"{o_car['nG']:,}", "AtN": f"{o_car['n']:,}",
    **{f"Sl{k}{kk}": fmt(v[0]) for k, d in slopes.items() for kk, v in d.items()},
    **{f"Sl{k}{kk}T": fmt(v[1]) for k, d in slopes.items() for kk, v in d.items()},
})
for (y, h, ex, pp, _), o in zip(specs, E):
    print(y, h, ex, pp, {k: (round(o[k], 2), round(o[k + "_t"], 2)) for k in ["D", "DG", "xD", "xDG", "x", "xG"]}, o["n"], o["nG"])
