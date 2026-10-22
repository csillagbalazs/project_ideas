"""Build the report-level sample: records, gaps, outcomes and controls.

Key identity. Trailing-twelve-month (TTM) EPS is the sum of the last four standalone quarters, so a
new report changes TTM EPS by exactly this quarter's EPS minus the same quarter a year earlier, the
seasonal-random-walk (SRW) error. In units of the firm's SD of past SRW errors:

    r   = (TTM_new - record) / sd            distance to the record after the report
    gap = (record - TTM_before) / sd >= 0    distance to the record before the report (known in advance)
    r   = SUE - gap                          a report sets a record when its surprise exceeds the gap.
"""
import numpy as np
import pandas as pd

from utils import DATA, macros, write
from common.data import announcements, monthly_panel, winsor
from common.features import abnormal_turnover

a = announcements().sort_values(["permno", "fyear", "fqtr"]).reset_index(drop=True)
a["name"] = (a.permno.str.replace(" - UNADJUSTED PRICE", "", regex=False)
             .str.replace("'A'", "", regex=False).str.replace(r"\s+DEAD.*$", "", regex=True).str.strip())
a["t"] = a.fyear * 4 + a.fqtr
g = a.groupby("permno", sort=False)


def consec(k):
    return (a.t - g.t.shift(k)) == k


a["ttm"] = g.eps_actual.transform(lambda s: s.rolling(4, min_periods=4).sum()).where(consec(3))
a["ttm_prev"] = g.ttm.shift(1).where(consec(1))
a["prevmax"] = g.ttm.transform(lambda s: s.shift(1).expanding(min_periods=8).max())
a["sue"] = a.e_rw / a.sd_ue                      # unwinsorised, so that r = sue - gap holds exactly
a["r"] = (a.ttm - a.prevmax) / a.sd_ue
a["gap"] = (a.prevmax - a.ttm_prev) / a.sd_ue
a.loc[a.gap.abs() < 1e-9, "gap"] = 0.0           # exact zero: the previous report was the record

# Quarters since the record was set (the latest quarter at which the running maximum was attained).
since = np.full(len(a), np.nan)
for _, idx in g.indices.items():
    best, best_t = -np.inf, np.nan
    for j in idx:
        since[j] = a.t.iat[j] - best_t
        v = a.ttm.iat[j]
        if np.isfinite(v) and v >= best:
            best, best_t = v, a.t.iat[j]
a["q_since"] = since

# Alternative record definitions (robustness): single quarter, same-period year-to-date, fiscal year.
a["max_q"] = g.eps_actual.transform(lambda s: s.shift(1).expanding(min_periods=8).max())
a["r_q"] = (a.eps_actual - a.max_q) / a.sd_ue
a["ytd"] = a.groupby(["permno", "fyear"]).eps_actual.cumsum()
nq = a.groupby(["permno", "fyear"]).fqtr.transform(lambda s: np.arange(1, len(s) + 1))
a.loc[nq != a.fqtr, "ytd"] = np.nan              # all earlier quarters of the year must be present
a["max_ytd"] = a.groupby(["permno", "fqtr"]).ytd.transform(lambda s: s.shift(1).expanding(min_periods=2).max())
a["r_ytd"] = (a.ytd - a.max_ytd) / a.sd_ue
a["r_fy"] = a.r_ytd.where(a.fqtr == 4)
for k in ["q", "ytd", "fy"]:
    a[f"gap_{k}"] = a.sue - a[f"r_{k}"]
a["z_ttm"] = a.ttm / a.sd_ue                     # distance to zero earnings (loss threshold)

# Neighbouring reports: previous and next surprise, earnings growth over the next four quarters.
a["prev_sue"] = g.sue_rw.shift(1).where(consec(1))
a["prev_streak"] = g.streak_rw.shift(1).where(consec(1))
a["next_sue"] = g.sue_rw.shift(-1).where((g.t.shift(-1) - a.t) == 1)
a["fut4"] = ((g.ttm.shift(-4) - a.ttm) / a.sd_ue).where((g.t.shift(-4) - a.t) == 4)

# Monthly outcomes and controls.
m = monthly_panel()
m["adjp"] = m.price / m.basis_factor             # price on the end-of-sample share basis
mm = m.set_index(["permno", "mi"])


def at(col, k):
    return mm[col].reindex(pd.MultiIndex.from_arrays([a.permno, a.ann_mi + k])).to_numpy()


a["car1_6"] = pd.DataFrame({k: at("aret", k) for k in range(1, 7)}).sum(axis=1, min_count=6).to_numpy()
high = np.nanmax(np.column_stack([at("adjp", -k) for k in range(12)]), axis=1)
a["pth"] = at("adjp", 0) / high                  # price relative to its 52-week high (start of month)
a["covered"] = np.isfinite(at("eps_12fwd", 0)).astype(float)
T = abnormal_turnover()
a["abto0"] = 100 * T.reindex(pd.MultiIndex.from_arrays([a.permno, a.ann_mi])).to_numpy()
a["rev"] = 1e4 * (at("eps_12fwd", 1) - at("eps_12fwd", 0)) / at("price", 0)   # basis points of price
a.loc[(a.ann_mi % 12) == 3, "rev"] = np.nan      # April reports: the I/B/E/S fiscal-year roll in May
a["annual"] = (a.fqtr == 4).astype(float)

# Sample selection.
flow = []
s = a[a.year.between(2005, 2020) & a.sue.notna()]
flow.append(("Reports with an SRW surprise, 2005--2020", len(s)))
s = s[s.ttm.notna() & s.ttm_prev.notna()]
flow.append(("Four consecutive quarters, now and one quarter earlier", len(s)))
s = s[s.prevmax.notna()]
flow.append(("At least eight earlier TTM values (record defined)", len(s)))
s = s[s.prevmax > 0]
flow.append(("Positive record", len(s)))
s = s[s.r != 0].copy()
flow.append(("Not an exact tie with the record (main sample)", len(s)))
assert (s.r - (s.sue - s.gap)).abs().max() < 1e-6, "identity r = sue - gap must hold"

for c in ["next_sue", "fut4", "rev", "abto0"]:
    s[c] = winsor(s[c], s.year)
for c in ["size", "bm", "mom", "pth"]:
    s[c + "_f"] = s.groupby("ann_mi")[c].transform(lambda v: v.fillna(v.median()))
s["rec"] = (s.r > 0).astype(float)
s["at_rec"] = (s.gap == 0).astype(float)
s["sbin"] = np.floor(s.sue / 0.25)
s["sbin10"] = np.floor(s.sue / 0.10)
s["sbin50"] = np.floor(s.sue / 0.50)
s["xbin"] = np.floor((s.e_xgb / s.sd_ue) / 0.25)
s["sue_pos"], s["sue_neg"] = s.sue_rw.clip(lower=0), s.sue_rw.clip(upper=0)
keep = ["permno", "name", "fyear", "fqtr", "ann_date", "ann_mi", "year", "eps_actual", "eps_lag4", "sd_ue",
        "ttm", "ttm_prev", "prevmax", "sue", "sue_rw", "sue_xgb", "sue_pos", "sue_neg", "r", "gap", "rec", "at_rec",
        "q_since", "sbin", "sbin10", "sbin50", "xbin", "prev_sue", "prev_streak", "streak_rw", "next_sue", "fut4",
        "ar0", "car1_3", "car1_6", "car_pre", "abto0", "rev", "size", "bm", "mom", "pth", "covered", "annual",
        "size_f", "bm_f", "mom_f", "pth_f", "me", "r_q", "r_ytd", "r_fy", "gap_q", "gap_ytd", "gap_fy", "z_ttm"]
s[keep].to_parquet(DATA / "sample.parquet")

# All reports (for the alternative record definitions and the zero-earnings benchmark).
allr = a[a.year.between(2005, 2020) & a.sue.notna()].copy()
for c in ["abto0", "next_sue"]:
    allr[c] = winsor(allr[c], allr.year)
allr["sbin"] = np.floor(allr.sue / 0.25)
allr[["permno", "fyear", "fqtr", "ann_mi", "year", "sue", "sbin", "r_q", "r_ytd", "r_fy", "gap_q", "gap_ytd",
      "gap_fy", "z_ttm", "eps_actual", "sd_ue", "ttm", "ar0", "car1_3", "abto0", "size"]].to_parquet(DATA / "reports_all.parquet")

lines = [f"{lab} & {n:,}\\\\" for lab, n in flow]
lines += [r"\midrule",
          f"New records ($r>0$) & {int(s.rec.sum()):,}\\\\",
          f"\\quad of which the previous report was the record ($gap=0$) & {int((s.rec * s.at_rec).sum()):,}\\\\",
          f"Firms & {s.permno.nunique():,}\\\\"]
write("tab_sample_flow.tex", "\\begin{tabular}{lr}\n\\toprule\nStep & Reports\\\\\n\\midrule\n" + "\n".join(lines)
      + "\n\\bottomrule\n\\end{tabular}\n")

near = s.r.abs() <= 1
macros("sample", {
    "NReports": f"{len(s):,}", "NFirms": f"{s.permno.nunique():,}", "NRecords": f"{int(s.rec.sum()):,}",
    "ShareRecords": f"{100 * s.rec.mean():.0f}",
    "ShareRecAtRec": f"{100 * s[s.rec == 1].at_rec.mean():.0f}",
    "ShareAtRec": f"{100 * s.at_rec.mean():.0f}",
    "ShareNearAtRec": f"{100 * s[near].at_rec.mean():.0f}",
    "NNear": f"{int(near.sum()):,}",
    "MedianGap": f"{s[s.gap > 0].gap.median():.1f}",
    "ShareRecFirms": f"{100 * s.groupby('permno').rec.max().mean():.0f}",
})
print(pd.DataFrame(flow, columns=["step", "n"]).to_string(index=False))
print("records", int(s.rec.sum()), "at-record share of records", round(s[s.rec == 1].at_rec.mean(), 3))
