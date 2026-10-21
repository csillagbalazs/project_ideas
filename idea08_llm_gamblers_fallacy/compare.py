"""Combine the cached Haiku and Sonnet results of run.py into one figure and table."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.plot import SERIES, INK2, plt, save  # noqa: E402

OUT = Path(__file__).parent / "results"
tabs = {m: pd.read_csv(OUT / f"by_streak_{m}.csv", index_col=0) for m in ["haiku", "sonnet"]}
slopes = pd.concat({m: pd.read_csv(OUT / f"slopes_{m}.csv", index_col=[0, 1]) for m in ["haiku", "sonnet"]})
acc = pd.concat({m: pd.read_csv(OUT / f"accuracy_{m}.csv", index_col=0).iloc[:, 0] for m in ["haiku", "sonnet"]}, axis=1)
print(slopes.round(3)); print(acc.round(3))

order = [-4, -3, -2, -1, 1, 2, 3, 4]
lbl = ["-4+", "-3", "-2", "-1", "+1", "+2", "+3", "+4+"]
t = tabs["haiku"].reindex(order)
fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.plot(range(8), 100 * t.realised_pop, marker="o", color=INK2, label="Realised frequency")
ax.plot(range(8), 100 * t.xgb_pop, marker="o", color=SERIES[2], label="XGBoost")
ax.plot(range(8), 100 * tabs["sonnet"].reindex(order).llm_p_up, marker="o", color=SERIES[0], label="LLM: Sonnet")
ax.plot(range(8), 100 * t.llm_p_up, marker="o", color=SERIES[1], label="LLM: Haiku")
ax.axhline(50, color=INK2, lw=0.6, ls=":")
ax.set_xticks(range(8))
ax.set_xticklabels(lbl)
ax.set_xlabel("Current streak of year-on-year EPS changes (sign x length)")
ax.set_ylabel("P(next quarter up YoY), %")
ax.set_title("Beliefs after earnings streaks: stated LLM probabilities vs outcomes")
ax.legend(loc="upper left", fontsize=7, ncol=2)
fig.tight_layout()
save(fig, OUT / "fig_compare.pdf")
slopes.to_csv(OUT / "slopes_both.csv")
acc.to_csv(OUT / "accuracy_both.csv")
