"""Shared panels for the project_ideas proofs of concept.

Timing conventions (verified on the raw file):
- A row dated the first of month m holds start-of-month values: price, the total
  return index (TRI), I/B/E/S consensus, and the volume traded in month m-1.
- ret (row m) = TRI(m+1)/TRI(m) - 1 is the return earned during calendar month m.
- turnover (row m) = volume/outstanding is month m-1 turnover, known at the start of m.

Scale conventions:
- Quarterly EPS in the forecast file is retroactively bonus-adjusted to the 2021
  share basis, while prices and I/B/E/S fields are on the share basis of their own
  date. Earnings surprises are therefore scaled by the firm's historical SD of
  seasonal-random-walk errors (basis-free), never by price.
- Shares outstanding is back-filled in places, so size and turnover are noisy for
  early years. This is a known caveat of the source panel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "clean"
CACHE = ROOT / "cache"
FORECASTS = ROOT.parent / "earnings_prediction" / "output" / "forecasts" / "eps_forecasts_oos.parquet"

MONTHLY_COLS = ["date", "permno", "exchange", "price", "volume", "outstanding", "totalreturn",
                "bookvalue", "eps_12fwd", "eps_12trl", "eps_12high", "eps_12low", "eps_12sd"]


def _mi(dates: pd.Series) -> pd.Series:
    """Integer month index (year*12 + month-1)."""
    return dates.dt.year * 12 + dates.dt.month - 1


def mi_to_date(mi) -> pd.Series:
    mi = pd.Series(mi)
    return pd.to_datetime(dict(year=mi // 12, month=mi % 12 + 1, day=1))


def winsor(s: pd.Series, by: pd.Series | None = None, p: float = 0.01) -> pd.Series:
    if by is None:
        lo, hi = s.quantile([p, 1 - p])
        return s.clip(lo, hi)
    lo = s.groupby(by).transform(lambda x: x.quantile(p))
    hi = s.groupby(by).transform(lambda x: x.quantile(1 - p))
    return s.clip(lo, hi)


def monthly_panel(refresh: bool = False) -> pd.DataFrame:
    """Firm-month panel with returns, controls and analyst fields."""
    path = CACHE / "monthly.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    d = pd.read_stata(CLEAN / "shanghai_shenzhen_monthly.dta", columns=MONTHLY_COLS)
    d = d.sort_values(["permno", "date"]).reset_index(drop=True)
    d["mi"] = _mi(d.date)
    g = d.groupby("permno", sort=False)
    nxt_tri, nxt_mi = g.totalreturn.shift(-1), g.mi.shift(-1)
    d["ret"] = np.where(nxt_mi == d.mi + 1, nxt_tri / d.totalreturn - 1, np.nan)
    d.loc[(d.ret > 2) | (d.totalreturn < 0.01), "ret"] = np.nan
    d["me"] = d.price * d.outstanding
    d["size"] = np.log(d.me)
    d["bm"] = np.log(d.bookvalue / d.price).where(d.bookvalue > 0)
    d["turnover"] = (d.volume / d.outstanding).clip(upper=3)
    # Lags use a dense month grid, so gaps never borrow a non-adjacent month.
    full = d.set_index(["permno", "mi"]).ret
    def lagged(k):
        s = full.copy()
        s.index = s.index.set_levels(s.index.levels[1] + k, level=1)
        return s.reindex(d.set_index(["permno", "mi"]).index).to_numpy()
    d["rev"] = lagged(1)
    logs = sum(np.log1p(pd.Series(lagged(k))) for k in range(2, 13))
    d["mom"] = np.expm1(logs.to_numpy())
    d["age"] = g.cumcount()
    # Share-basis factor: product of bonus/split events from month m to the end of the
    # sample. A month whose TRI-to-price growth ratio exceeds 1.05 is treated as a share
    # event (cash dividends rarely exceed 5% in one month). Multiplying the 2021-basis
    # adjusted EPS by this factor restores the share basis of month m.
    ratio = (nxt_tri / d.totalreturn) / (g.price.shift(-1) / d.price)
    d["share_event"] = ratio.where(ratio > 1.05, 1.0).fillna(1.0)
    d["basis_factor"] = d.iloc[::-1].groupby("permno", sort=False).share_event.cumprod().iloc[::-1]
    # Size-decile benchmark (equal-weighted) for abnormal returns.
    ok = d.ret.notna() & d["size"].notna()
    d["size_dec"] = np.nan
    d.loc[ok, "size_dec"] = d[ok].groupby("mi")["size"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    bench = d[ok].groupby(["mi", "size_dec"]).ret.mean().rename("ret_bench")
    d = d.merge(bench, on=["mi", "size_dec"], how="left")
    d["aret"] = d.ret - d.ret_bench
    mkt = d[ok].groupby("mi").apply(lambda t: np.average(t.ret, weights=t.me.clip(lower=0) + 1e-12),
                                    include_groups=False).rename("mkt_vw")
    d = d.merge(mkt, on="mi", how="left")
    CACHE.mkdir(exist_ok=True)
    d.to_parquet(path)
    return d


def factors() -> pd.DataFrame:
    """Liu-Stambaugh-Yuan CH-3/CH-4 factors in decimals, keyed by month index."""
    f = pd.read_stata(CLEAN / "china_factors.dta")
    f["mi"] = _mi(f.date)
    for c in ["rf_mon", "mktrf", "VMG", "SMB", "PMO"]:
        f[c] = f[c] / 100
    return f[["mi", "rf_mon", "mktrf", "SMB", "VMG", "PMO"]]


def announcements(refresh: bool = False) -> pd.DataFrame:
    """One row per firm-fiscal-quarter announcement, with surprises and returns.

    SUE_m = (EPS - forecast_m) / SD of the firm's last 8 SRW errors published before
    the announcement (min 4). All models share this denominator, so their scales match.
    """
    path = CACHE / "announcements.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    cols = ["permno", "fyear", "fqtr", "ann_date", "date_flag", "eps_actual", "eps_lag4",
            "f_rw", "f_xgb", "e_rw", "e_xgb", "avail_rw", "avail_xgb", "common", "lag_days"]
    a = pd.read_parquet(FORECASTS, columns=cols)
    a = a[a.ann_date.notna()].sort_values(["permno", "ann_date", "fyear", "fqtr"]).reset_index(drop=True)
    # Historical SD of SRW errors that were public before this announcement.
    sds = np.full(len(a), np.nan)
    for _, idx in a.groupby("permno", sort=False).indices.items():
        e = a.e_rw.to_numpy()[idx]
        dts = a.ann_date.to_numpy()[idx]
        for j in range(len(idx)):
            past = e[:j][(dts[:j] < dts[j])]
            past = past[~np.isnan(past)][-8:]
            if len(past) >= 4:
                sds[idx[j]] = np.std(past, ddof=1)
    a["sd_ue"] = sds
    a.loc[a.sd_ue <= 0, "sd_ue"] = np.nan
    a["sue_rw"] = a.e_rw / a.sd_ue
    a["sue_xgb"] = a.e_xgb / a.sd_ue
    a["ann_mi"] = _mi(a.ann_date)
    a["year"] = a.ann_date.dt.year
    for c in ["sue_rw", "sue_xgb"]:
        a[c] = winsor(a[c], a.year)
    # SRW-based streaks (consecutive same-sign SRW surprises, in announcement order).
    sign = np.sign(a.e_rw)
    streak = np.zeros(len(a))
    for _, idx in a.groupby("permno", sort=False).indices.items():
        s, run = sign.to_numpy()[idx], 0
        for j, v in enumerate(s):
            if np.isnan(v) or v == 0:
                run = 0
            elif run != 0 and np.sign(run) == v:
                run += v
            else:
                run = v
            streak[idx[j]] = run
    a["streak_rw"] = streak

    m = monthly_panel()[["permno", "mi", "ret", "aret", "size", "bm", "turnover", "mom", "rev",
                         "price", "me", "eps_12fwd", "eps_12sd"]]
    mm = m.set_index(["permno", "mi"])
    def at(offset, col):
        key = pd.MultiIndex.from_arrays([a.permno, a.ann_mi + offset])
        return mm[col].reindex(key).to_numpy()
    for c in ["size", "bm", "turnover", "mom", "price", "me", "eps_12fwd", "eps_12sd"]:
        a[c] = at(0, c)
    a["r0"], a["ar0"] = at(0, "ret"), at(0, "aret")
    for k in (1, 2, 3):
        a[f"ar{k}"] = at(k, "aret")
    a["car1_3"] = a[["ar1", "ar2", "ar3"]].sum(axis=1, min_count=3)
    a["car_pre"] = pd.DataFrame({k: at(-k, "aret") for k in (1, 2)}).sum(axis=1, min_count=2)
    CACHE.mkdir(exist_ok=True)
    a.to_parquet(path)
    return a
