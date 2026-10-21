"""Firm-month features from the announcement history and the I/B/E/S consensus."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import announcements, mi_to_date, monthly_panel


def last_announcement_features() -> pd.DataFrame:
    """Per announcement: TTM EPS growth and the latest surprises, keyed by ann_date."""
    a = announcements().sort_values(["permno", "fyear", "fqtr"]).copy()
    g = a.groupby("permno", sort=False)
    a["ttm"] = g.eps_actual.transform(lambda s: s.rolling(4, min_periods=4).sum())
    a["ttm_l4"] = a.groupby("permno", sort=False).ttm.shift(4)
    a["ttm_growth"] = ((a.ttm - a.ttm_l4) / a.ttm_l4.abs()).clip(-3, 3)
    a["pos_share4"] = g.e_rw.transform(lambda s: (s > 0).astype(float).rolling(4, min_periods=2).mean())
    a["roe_sign"] = np.sign(a.ttm)
    return a[["permno", "ann_date", "sue_rw", "sue_xgb", "streak_rw", "ttm_growth", "pos_share4",
              "roe_sign", "ttm", "fqtr"]].sort_values("ann_date")


def attach_last_announcement(m: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: the latest announcement published strictly before the month starts."""
    f = last_announcement_features().dropna(subset=["ann_date"])
    m = m.copy()
    m["month_start"] = mi_to_date(m.mi).to_numpy()
    m = m.sort_values("month_start")
    out = pd.merge_asof(m, f, left_on="month_start", right_on="ann_date", by="permno",
                        allow_exact_matches=False, direction="backward")
    if "basis_factor" in out:
        out["ttm_asof"] = out.ttm * out.basis_factor  # realised TTM EPS on month-m share basis
    return out.sort_values(["permno", "mi"]).reset_index(drop=True)


def analyst_panel() -> pd.DataFrame:
    """Covered firm-months with price-scaled consensus fields and the 12-month-ahead outcome.

    The realised outcome stays inside I/B/E/S: trl_{m+12} (trailing EPS twelve months later),
    so share bases match the forecast and the price. FE = (trl_{m+12} - fwd_m) / P_m.
    """
    m = monthly_panel()
    idx = m.set_index(["permno", "mi"])
    def lead(col, k):
        key = pd.MultiIndex.from_arrays([m.permno, m.mi + k])
        return idx[col].reindex(key).to_numpy()
    m["trl_f12"] = lead("eps_12trl", 12)
    m["fwd_l1"], m["fwd_l3"] = lead("eps_12fwd", -1), lead("eps_12fwd", -3)
    m["price_l3"] = lead("price", -3)
    p = m.price
    m["F"] = m.eps_12fwd / p
    m["T"] = m.eps_12trl / p
    m["R"] = m.trl_f12 / p
    m["FE"] = m.R - m.F
    m["disp"] = m.eps_12sd / p
    m["range"] = (m.eps_12high - m.eps_12low) / p
    m["rev1"] = (m.eps_12fwd - m.fwd_l1) / p
    m["rev3"] = (m.eps_12fwd - m.fwd_l3) / p
    m["fgrowth"] = (m.eps_12fwd - m.eps_12trl) / p
    m["bp"] = np.exp(m.bm)
    m = m[m.eps_12fwd.notna() & (m.price > 0)]
    return attach_last_announcement(m)
