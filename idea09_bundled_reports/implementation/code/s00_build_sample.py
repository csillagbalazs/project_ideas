"""Build the firm-year sample: one annual report (fiscal year Y) paired with the Q1 report of Y+1.

Inputs come from the shared project_ideas pipeline (common/data.py):
  - announcement panel: EPS, publication dates, SRW and XGBoost forecasts, surprises,
  - monthly panel: size-adjusted returns, turnover, size, book-to-market, momentum.
Output: output/data/sample.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # .../project_ideas
sys.path.insert(0, str(ROOT))
from common.data import announcements, monthly_panel  # noqa: E402
from utils import DATA  # noqa: E402

a = announcements()
m = monthly_panel()

# ---------------------------------------------------------------- monthly helpers
m = m.sort_values(["permno", "mi"])
# Row m holds the turnover of month m-1; re-index so that turnover is dated by the month traded.
to = m[["permno", "mi", "turnover"]].copy()
to["mi"] = to.mi - 1
to["lto"] = np.log(to.turnover.where(to.turnover > 0))
to = to.set_index(["permno", "mi"]).lto
aret = m.set_index(["permno", "mi"]).aret
ret_full = aret.unstack()  # firm x month matrix (wide) for fast window sums
to_full = to.unstack()


def window_sum(permno: pd.Series, start: pd.Series, end: pd.Series, mat: pd.DataFrame, min_frac=1.0) -> np.ndarray:
    """Sum of mat[permno, start..end] (inclusive); NaN if fewer than min_frac of months observed."""
    cols = {c: i for i, c in enumerate(mat.columns)}
    rows = {p: i for i, p in enumerate(mat.index)}
    arr = mat.to_numpy()
    out = np.full(len(permno), np.nan)
    for k, (p, s, e) in enumerate(zip(permno, start, end)):
        if p not in rows or pd.isna(s) or pd.isna(e) or s not in cols or e not in cols or e < s:
            continue
        v = arr[rows[p], cols[s]:cols[e] + 1]
        if np.isfinite(v).mean() >= min_frac:
            out[k] = np.nansum(v)
    return out


def abnormal_turnover(permno, month):
    """log turnover in `month` minus the firm's mean log turnover over months-12..-2."""
    cols = {c: i for i, c in enumerate(to_full.columns)}
    rows = {p: i for i, p in enumerate(to_full.index)}
    arr = to_full.to_numpy()
    out = np.full(len(permno), np.nan)
    for k, (p, mm) in enumerate(zip(permno, month)):
        if p not in rows or mm not in cols or cols[mm] < 12:
            continue
        r = arr[rows[p]]
        base = r[cols[mm] - 12:cols[mm] - 1]
        if np.isfinite(r[cols[mm]]) and np.isfinite(base).sum() >= 6:
            out[k] = r[cols[mm]] - np.nanmean(base)
    return out


# ---------------------------------------------------------------- pair annual and Q1 reports
cols = ["permno", "fyear", "ann_date", "ann_mi", "date_flag", "eps_actual", "e_rw", "e_xgb", "sd_ue",
        "sue_rw", "sue_xgb", "size", "bm", "mom", "turnover", "eps_12fwd", "lag_days"]
q4 = a[a.fqtr == 4][cols]
q1 = a[a.fqtr == 1][cols].assign(fyear=lambda d: d.fyear - 1)
s = q4.merge(q1, on=["permno", "fyear"], suffixes=("4", "1"))
eps = a.pivot_table(index=["permno", "fyear"], columns="fqtr", values="eps_actual")
eps.columns = [f"eps_q{int(c)}" for c in eps.columns]
s = s.join(eps, on=["permno", "fyear"])

n0 = len(s)
s = s[s.fyear.between(2005, 2019)]
n1 = len(s)
s["gap"] = (s.ann_date1 - s.ann_date4).dt.days
s = s[s.gap >= 0]          # the Q1 report may not precede the annual report (CSRC rule)
n2 = len(s)
s = s[s.gap <= 150]
n3 = len(s)
s = s.dropna(subset=["sue_rw4", "sue_rw1"])
n4 = len(s)

s["same_day"] = (s.gap == 0).astype(int)
s["same_month"] = ((s.gap > 0) & (s.ann_mi1 == s.ann_mi4)).astype(int)
s["separate"] = (s.ann_mi1 > s.ann_mi4).astype(int)
s["cls"] = np.select([s.same_day == 1, s.same_month == 1], ["Same day", "Same month"], "Separate months")

# Timing of the annual report
fye = pd.to_datetime((s.fyear).astype(str) + "-12-31")
s["lag4"] = (s.ann_date4 - fye).dt.days            # days after fiscal year end
s["late_rank"] = s.groupby("fyear").lag4.rank(pct=True)
s["april4"] = (s.ann_date4.dt.month == 4).astype(int)
s["last10"] = ((s.ann_date4.dt.month == 4) & (s.ann_date4.dt.day >= 21)).astype(int)
s["d4"] = s.ann_date4.dt.strftime("%Y-%m-%d")       # annual-report date (fixed-effect key)
s["n_same_date"] = s.groupby("d4").permno.transform("size")  # cross-firm crowding on that date

# News variables
s["s4"], s["s1"] = s.sue_rw4, s.sue_rw1                   # SRW surprises (primary)
s["x4"], s["x1"] = s.sue_xgb4, s.sue_xgb1                 # XGBoost surprises (robustness)
s["neg4"] = (s.s4 < 0).astype(int)
s["neg1"] = (s.s1 < 0).astype(int)
s["fy_eps"] = s[["eps_q1", "eps_q2", "eps_q3", "eps_q4"]].sum(axis=1, min_count=4)
s["loss"] = (s.fy_eps < 0).astype(int)
s["covered"] = s.eps_12fwd4.notna().astype(int)
s["size"] = s.size4
s["bm"] = s.bm4
s["mom"] = s.mom4

# Returns (sums of monthly size-adjusted returns)
Y1 = (s.fyear + 1) * 12                                    # January of the reporting year
s["ar_m4"] = window_sum(s.permno, s.ann_mi4, s.ann_mi4, ret_full)
s["ar_m1"] = window_sum(s.permno, s.ann_mi1, s.ann_mi1, ret_full)
s["car_win"] = window_sum(s.permno, s.ann_mi4, s.ann_mi1, ret_full)        # both releases
s["car_janapr"] = window_sum(s.permno, Y1, Y1 + 3, ret_full)               # same calendar window for all
s["car_pre"] = window_sum(s.permno, Y1, s.ann_mi4 - 1, ret_full)           # Jan .. month before annual
s.loc[s.ann_mi4 == Y1, "car_pre"] = 0.0
s["car_post3"] = window_sum(s.permno, s.ann_mi1 + 1, s.ann_mi1 + 3, ret_full)
s["car_post6"] = window_sum(s.permno, s.ann_mi1 + 1, s.ann_mi1 + 6, ret_full)
s["car_mayjul"] = window_sum(s.permno, Y1 + 4, Y1 + 6, ret_full)
s["car_m4_next"] = window_sum(s.permno, s.ann_mi4, s.ann_mi4 + 1, ret_full)  # release month + next month
for k in range(-3, 4):                                                       # event-time months around m4
    s[f"r{k:+d}"] = window_sum(s.permno, s.ann_mi4 + k, s.ann_mi4 + k, ret_full)
s["both_same_month"] = (s.ann_mi1 == s.ann_mi4).astype(int)
s["abto_m4"] = abnormal_turnover(s.permno, s.ann_mi4)
s["abto_m1"] = abnormal_turnover(s.permno, s.ann_mi1)

# Firm history of bundling
s = s.sort_values(["permno", "fyear"])
prev = s.groupby("permno").same_day.shift(1)
s["bundled_prev"] = prev.where(s.groupby("permno").fyear.diff() == 1)
s["firm"] = pd.factorize(s.permno)[0]

s.to_parquet(DATA / "sample.parquet")
flow = pd.DataFrame({"step": ["Annual--Q1 report pairs", "Fiscal years 2005--2019", "Q1 not before annual report",
                              "Gap at most 150 days", "Both surprises available"],
                     "n": [n0, n1, n2, n3, n4]})
flow = flow[flow.n.diff().fillna(-1) != 0]  # report only steps that remove observations
flow.to_csv(DATA / "sample_flow.csv", index=False)
print(flow.to_string(index=False))
print(s.cls.value_counts().to_dict())
print(s[["ar_m4", "car_win", "car_janapr", "car_post3", "abto_m4"]].describe().T.round(3))
