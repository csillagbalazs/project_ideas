"""Run the full pipeline: sample -> descriptives -> naive RD -> moving cutoff -> at-record firms -> bunching -> robustness."""
import runpy
from pathlib import Path

HERE = Path(__file__).parent
for script in ["s00_build_sample.py", "s01_descriptives.py", "s02_naive.py", "s03_moving_cutoff.py",
               "s04_at_record.py", "s05_bunching.py", "s06_robustness.py"]:
    print(f"\n=== {script} ===")
    runpy.run_path(str(HERE / script), run_name="__main__")
