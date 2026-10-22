# Breaking Records: Are Record Earnings a Reference Point for Investors?

The paper is `paper/paper.pdf` (source: `paper/paper.tex`). Every table, figure and number in the text is produced by
the code below.

## Run

```bash
cd project_ideas/idea03_record_earnings/implementation/code
/opt/homebrew/anaconda3/envs/eps-forecast/bin/python run_all.py      # under a minute
cd ../paper && tectonic -X compile paper.tex
```

The scripts reuse the shared panels in `project_ideas/common/data.py`. Those panels are built from `project_ideas/clean/`
and the out-of-sample forecasts in `earnings_prediction/output/forecasts/`.

## Code (`code/`)

| Script | Paper section | Outputs |
|---|---|---|
| `s00_build_sample.py` | 3 Data | `output/data/sample.parquet`, `reports_all.parquet`, Table 1 |
| `s01_descriptives.py` | 3 | Table 2, Figures 1-2 (Wuliangye example, record frequency, the gap) |
| `s02_naive.py` | 5.1 Naive RD (the proof-of-concept design) | Table 3, Figure 3 |
| `s03_moving_cutoff.py` | 5.2 Main result | Tables 4-5, Figures 4-5 |
| `s04_at_record.py` | 5.3 Firms at their record (difference in discontinuities) | Table 6, Figure 6 |
| `s05_bunching.py` | 5.4 Earnings management at the record | Table 7, Figure 7 |
| `s06_robustness.py` | 5.5 Robustness, other record definitions, placebos | Table A1, Figure 8 |
| `utils.py` | | RD with absorbed fixed effects, difference in discontinuities, Poisson bunching test, LaTeX output |

Numbers quoted in the text are written to `output/tables/macros_*.tex` and loaded by the paper.

## Main design in one paragraph

Trailing twelve-month EPS changes in each report by exactly the seasonal-random-walk error, so the distance to the
record after a report is `r = SUE - gap`, where `gap` is the distance to the record before the report (known in
advance). 72% of records are set by firms already at their record (`gap = 0`), for which a record is the same event as
beating last year's quarter. The main design therefore uses firms below their record and gives every 0.25-SD surprise
bin its own intercept. The jump at `r = 0` then compares firms with the same news that land on different sides of their
record only because their pre-report gap differed. Standard errors are two-way clustered by report month and firm.

## Relation to the proof of concept

The PoC (`../run.py`, `../summary.pdf`) found a kink in the three-month drift at the record. The paper shows that this
kink comes from firms at their record, where `r = SUE`, and that the same naive design finds kinks in firm size and the
previous surprise. In the moving-cutoff design there is no jump in returns, turnover, analyst revisions or future
earnings, and no bunching at the record.
