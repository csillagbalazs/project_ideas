# Two Reports, One Day: Do Investors Neglect Bundled Earnings News?

The paper is `paper/paper.pdf` (source: `paper/paper.tex`). Every table, figure and number in the text is produced by
the code below.

## Run

```bash
cd project_ideas/idea09_bundled_reports/implementation/code
/opt/homebrew/anaconda3/envs/eps-forecast/bin/python run_all.py      # about a minute
cd ../paper && tectonic -X compile paper.tex
```

The scripts reuse the shared panels in `project_ideas/common/data.py`. Those panels are built from `project_ideas/clean/`
and the out-of-sample forecasts in `earnings_prediction/output/forecasts/`.

## Code (`code/`)

| Script | Paper section | Outputs |
|---|---|---|
| `s00_build_sample.py` | Data | `output/data/sample.parquet`, sample flow |
| `s01_descriptives.py` | 3 | Tables 1-2, Figures 1-2 |
| `s02_selection.py` | 5.1 Who bundles? | Table 3, Figure 3 |
| `s03_naive_benchmark.py` | 5.2 Naive gap and information-set benchmark | Table 4, Figure 4 |
| `s04_same_date.py` | 5.3 Main result (same filing date design) | Table 5, Figures 5 and coefficient plot |
| `s05_volume.py` | 5.4 Trading volume | Table 6, Figure 6 |
| `s06_robustness.py` | 5.5 Heterogeneity, robustness, burying, placebo | Table 7, Figure 7 |
| `utils.py` | | fixed-effect absorption, group-specific slopes (FWL), two-way clustering, LaTeX output |

Numbers quoted in the text are written to `output/tables/macros_*.tex` and loaded by the paper.

## Main design in one paragraph

Firm-years whose annual and Q1 reports fall in the same calendar month are split into *bundles* (same day) and
*same-month* releases (a few days apart). Each bundle is compared only with firms that filed their annual report on
the same date. Every date (and, in other specifications, every firm) gets its own intercept and its own response to
both earnings surprises. Standard errors are two-way clustered by date and firm.
