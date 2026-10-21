# Project ideas: proofs of concept

`research_ideas.tex` lists the candidate questions. The last column summarises the proof-of-concept
(PoC) results for Sections B and C. Each tested idea has its own folder:

| Folder | Idea |
|---|---|
| `idea04_whose_expectations` | Whose earnings expectations do prices reflect (SRW vs XGBoost)? |
| `idea05_ml_analyst_bias` | Machine benchmark for Chinese analysts: conditional bias and returns |
| `idea06_sticky_expectations` | Sticky analyst expectations and the profitability anomaly |
| `idea07_q4_dip` | The fourth-quarter earnings dip |
| `idea08_llm_gamblers_fallacy` | Do LLM forecasters show the gambler's fallacy? |
| `idea09_bundled_reports` | Bundled annual and Q1 reports (limited attention) |
| `idea10_disposition_cgo` | Disposition effect (capital-gains overhang) and earnings news |
| `idea11_new_investors` | Waves of new investors and earnings news |

Each folder contains `run.py` (the analysis), `results/` (CSV outputs and figures) and `summary.tex`/`summary.pdf`
(the one-page summary).

## Running

Use the `eps-forecast` conda environment (it has pandas, statsmodels, xgboost and matplotlib):

```bash
cd project_ideas
PY=/opt/homebrew/anaconda3/envs/eps-forecast/bin/python
$PY idea04_whose_expectations/run.py          # any idea folder
$PY idea09_bundled_reports/run.py sue_rw      # Idea 9 also takes sue_xgb (default) or sue_rw
$PY idea08_llm_gamblers_fallacy/run.py haiku  # Idea 8 takes a model alias; LLM answers are cached in results/raw/
cd idea04_whose_expectations && tectonic -X compile summary.tex
```

## Shared code (`common/`)

- `data.py` builds and caches (in `cache/`) the firm-month panel and the announcement panel from `clean/` and the
  out-of-sample forecasts in `../earnings_prediction/output/forecasts/eps_forecasts_oos.parquet`.
- `features.py` builds the analyst panel and the as-of merge of the latest announcement.
- `stats.py` holds Fama-MacBeth, clustered OLS, portfolio sorts and factor alphas (Liu-Stambaugh-Yuan CH-3/CH-4).
- `plot.py` sets the figure style.

Delete `cache/` (or call the builders with `refresh=True`) after changing `data.py`.

## Data conventions worth knowing

- A row dated the first of month *m* holds start-of-month prices and I/B/E/S values and the volume of month *m-1*.
  `ret` is the return earned during month *m*.
- Quarterly EPS is bonus-adjusted to the 2021 share basis, while prices and I/B/E/S fields use the share basis of
  their own date. Surprises are therefore scaled by the SD of past SRW errors, not by price. `basis_factor`
  (inferred from total-return-index vs price jumps) converts adjusted EPS back to the as-of basis.
- `eps_12fwd`/`eps_12trl` are time-weighted I/B/E/S 12-month blends, not fiscal-year values. Analyst tests stay
  inside I/B/E/S (realised outcome = trailing EPS twelve months later).
- Shares outstanding is back-filled in places, so size and turnover are noisier in early years.
- Idea 11 uses CSDC new-investor counts downloaded from Eastmoney (`idea11_new_investors/data_raw/`, 2015-04 onward).
