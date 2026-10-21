"""Idea 8 PoC: do LLM earnings forecasters show the gambler's fallacy?

Each case is an anonymised history of 12 standalone quarterly EPS (oldest first, labelled
by fiscal quarter only, rescaled by a firm-specific constant, so neither the firm nor the
period can be recognised). The LLM forecasts the next quarter and gives the probability that
it exceeds the same quarter one year earlier (p_up). Cases are stratified by the streak of
year-on-year EPS changes that ends at the last observed quarter: +1..+4 (4 = four or more)
and -1..-4.

Gambler's fallacy: p_up falls with the length of an up-streak (and rises with a down-streak)
by more than the realised frequencies do. Hot hand: the opposite.

LLM calls go through the local `claude` CLI (headless, no tools) and are cached in
results/raw/, so reruns do not call the model again.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import announcements  # noqa: E402
from common.stats import ols_cluster  # noqa: E402
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "haiku"
PER_BUCKET, BATCH, HIST = 40, 20, 12
WORKDIR = "/private/tmp/claude-501/-Users-csillagb-Documents-personal/e8b28bb7-0ccc-4b2b-9992-3a4b1e031c67/scratchpad"

# ---------- build cases ----------
a = announcements().sort_values(["permno", "fyear", "fqtr"]).reset_index(drop=True)
a["t"] = a.fyear * 4 + a.fqtr - 1
cases = []
for p, g in a.groupby("permno", sort=False):
    t, eps = g.t.to_numpy(), g.eps_actual.to_numpy()
    for j in range(HIST + 4, len(g)):
        if g.fyear.iloc[j] < 2012 or g.fyear.iloc[j] > 2019:
            continue
        win = slice(j - HIST, j)
        if t[j] - t[j - HIST] != HIST or np.isnan(eps[j - HIST - 4:j + 1]).any():
            continue
        # streak of YoY changes ending at the last observed quarter j-1
        yoy = np.sign(eps[j - HIST:j] - eps[j - HIST - 4:j - 4])
        last = yoy[-1]
        if last == 0:
            continue
        k = 0
        for v in yoy[::-1]:
            if v == last:
                k += 1
            else:
                break
        cases.append({"permno": p, "fyear": g.fyear.iloc[j], "fqtr": g.fqtr.iloc[j],
                      "hist": eps[win].tolist(), "hist_q": g.fqtr.iloc[win].tolist(),
                      "target": eps[j], "lag4": eps[j - 4], "f_xgb": g.f_xgb.iloc[j],
                      "f_rw": g.f_rw.iloc[j], "streak": int(last * min(k, 4))})
c = pd.DataFrame(cases)
c = c[c.target != c.lag4]
print("eligible cases:", len(c), c.streak.value_counts().sort_index().to_dict())

# Realised base rates on the full eligible population (not just the LLM sample).
c["up"] = (c.target > c.lag4).astype(float)
c["xgb_up"] = (c.f_xgb > c.lag4).astype(float).where(c.f_xgb.notna())
base = c.groupby("streak")[["up", "xgb_up"]].mean()
print(base.round(3))

rng = np.random.default_rng(7)
s = (c.groupby("streak", group_keys=False)
      .apply(lambda g: g.sample(min(PER_BUCKET, len(g)), random_state=7))
      .sample(frac=1, random_state=7).reset_index(drop=True))
s["id"] = np.arange(len(s))
s["scale"] = s["hist"].apply(lambda h: np.mean(np.abs(h)) or 1.0) / rng.uniform(80, 120, len(s))


def fmt(row):
    vals = [v / row.scale for v in row.hist]
    return "; ".join(f"Q{q}: {v:.1f}" for q, v in zip(row.hist_q, vals))


PROMPT = """You will forecast company earnings. Each series below lists one company's quarterly
earnings per share for the last 12 quarters, oldest first, in arbitrary units. Labels are
fiscal quarters (Q1..Q4); earnings can be seasonal. For each series give:
- "forecast": your forecast of the NEXT quarter's value (same units),
- "p_up": your probability (0-1) that the next quarter's value will be higher than the value
  of the same fiscal quarter one year earlier (four quarters before the next quarter).
Answer with only a JSON array of objects {"id": int, "forecast": number, "p_up": number}.

"""


def call(batch_id, rows):
    path = RAW / f"{MODEL}_batch{batch_id:03d}.json"
    if path.exists():
        return json.loads(path.read_text())
    body = PROMPT + "\n".join(f"id {r.id}: {fmt(r)}" for r in rows.itertuples())
    cmd = ["claude", "-p", "--model", MODEL, "--output-format", "json", "--tools", "",
           "--system-prompt", "You are a careful quantitative earnings forecaster. Reply with JSON only."]
    out = subprocess.run(cmd, input=body, capture_output=True, text=True, cwd=WORKDIR, timeout=600)
    res = json.loads(out.stdout)
    txt = res["result"].strip()
    txt = txt[txt.index("["): txt.rindex("]") + 1]
    parsed = {"items": json.loads(txt), "cost": res.get("total_cost_usd")}
    path.write_text(json.dumps(parsed))
    return parsed


batches = [s.iloc[i:i + BATCH] for i in range(0, len(s), BATCH)]
with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(lambda ib: call(*ib), enumerate(batches)))
cost = sum(r.get("cost") or 0 for r in results)
llm = pd.DataFrame([it for r in results for it in r["items"]])
s = s.merge(llm, on="id", how="left")
s["llm_forecast"] = s.forecast * s.scale
s["llm_up"] = (s.llm_forecast > s.lag4).astype(float)
print(f"model={MODEL} cases={len(s)} answered={s.p_up.notna().sum()} cost=${cost:.2f}")

# ---------- analysis ----------
tab = s.groupby("streak").agg(n=("id", "size"), llm_p_up=("p_up", "mean"), llm_up=("llm_up", "mean"),
                              realised=("up", "mean"), xgb_up=("xgb_up", "mean"))
tab = tab.join(base.rename(columns={"up": "realised_pop", "xgb_up": "xgb_pop"}))
print(tab.round(3))
tab.to_csv(OUT / f"by_streak_{MODEL}.csv")

# Slope of beliefs on streak length within up- and down-streaks, relative to realised outcomes.
s["k"] = s.streak.abs()
s["dir"] = np.sign(s.streak)
s["belief_minus_real"] = s.p_up - s.up
reg = {}
for d_, lab in [(1, "up-streaks"), (-1, "down-streaks")]:
    g = s[s.dir == d_].copy()
    g["one"] = 1
    for y in ["p_up", "up", "xgb_up"]:
        r = ols_cluster(g, y, ["k"], cluster="permno")
        reg[(lab, y)] = (r.params["k"], r.tvalues["k"])
print({k: (round(v[0], 3), round(v[1], 2)) for k, v in reg.items()})
pd.DataFrame(reg, index=["slope", "t"]).T.to_csv(OUT / f"slopes_{MODEL}.csv")

# Accuracy in scaled units (|error| / scale), common rows.
e = s.dropna(subset=["llm_forecast", "f_xgb", "f_rw"])
acc = {m: ((e.target - e[f]).abs() / e.scale).median() for m, f in
       [("LLM", "llm_forecast"), ("XGBoost", "f_xgb"), ("SRW", "f_rw")]}
dir_acc = {"LLM": (e.llm_up == e.up).mean(), "XGBoost": (e.xgb_up == e.up).mean(),
           "Brier LLM": ((e.p_up - e.up) ** 2).mean()}
print("median |error| (scaled):", {k: round(v, 3) for k, v in acc.items()}, "direction:", dir_acc)
pd.Series({**{f"mae_{k}": v for k, v in acc.items()}, **dir_acc, "cost": cost, "n": len(s)}).to_csv(
    OUT / f"accuracy_{MODEL}.csv")
s.drop(columns=["hist", "hist_q"]).to_csv(OUT / f"cases_{MODEL}.csv", index=False)

# Figure: probability of an increase by streak.
order = [-4, -3, -2, -1, 1, 2, 3, 4]
lbl = ["-4+", "-3", "-2", "-1", "+1", "+2", "+3", "+4+"]
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.plot(range(8), 100 * tab.reindex(order).realised_pop, marker="o", color=INK2, label="Realised frequency (all eligible cases)")
ax.plot(range(8), 100 * tab.reindex(order).xgb_pop, marker="o", color=SERIES[1], label="XGBoost forecast")
ax.plot(range(8), 100 * tab.reindex(order).llm_p_up, marker="o", color=SERIES[0], label=f"LLM ({MODEL}) stated probability")
ax.set_xticks(range(8))
ax.set_xticklabels(lbl)
ax.set_xlabel("Current streak of year-on-year EPS changes (sign x length)")
ax.set_ylabel("P(next quarter up YoY), %")
ax.set_title("Beliefs after streaks: LLM vs realised outcomes")
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout()
save(fig, OUT / f"fig_main_{MODEL}.pdf")
