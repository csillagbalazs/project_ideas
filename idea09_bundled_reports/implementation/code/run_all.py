"""Run the full pipeline: sample -> descriptives -> selection -> benchmark -> main -> volume -> robustness."""
import runpy
from pathlib import Path

HERE = Path(__file__).parent
for script in ["s00_build_sample.py", "s01_descriptives.py", "s02_selection.py", "s03_naive_benchmark.py",
               "s04_same_date.py", "s05_volume.py", "s06_robustness.py"]:
    print(f"\n=== {script} ===")
    runpy.run_path(str(HERE / script), run_name="__main__")
