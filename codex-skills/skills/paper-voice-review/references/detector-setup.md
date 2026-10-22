# Detector setup and execution

The review uses [`csillagbalazs/econ-ai-detector`](https://github.com/csillagbalazs/econ-ai-detector), pinned to commit `11285447a3ccdc300bef25bc0a4a1eb42fd489cb` when reproducibility matters.

## First-use setup

Prefer an existing Python 3.10+ environment in which `econ_ai_detector` imports successfully. Otherwise, request approval before downloading packages and create an isolated virtual environment under the Codex cache directory, not inside the manuscript repository. Do not use macOS's Python 3.9 system interpreter; the detector declares Python 3.10 or newer. Install the pinned source:

```bash
<python-3.10-or-newer> -m venv <codex-cache>/paper-voice-review/venv
<codex-cache>/paper-voice-review/venv/bin/python -m pip install "git+https://github.com/csillagbalazs/econ-ai-detector.git@11285447a3ccdc300bef25bc0a4a1eb42fd489cb"
```

If no suitable interpreter is installed but `uv` is available, an isolated managed interpreter is also acceptable:

```bash
uv venv --python 3.11 <codex-cache>/paper-voice-review/venv
<codex-cache>/paper-voice-review/venv/bin/python -m pip install "git+https://github.com/csillagbalazs/econ-ai-detector.git@11285447a3ccdc300bef25bc0a4a1eb42fd489cb"
```

The DistilRoBERTa weights are approximately 330 MB and download from Hugging Face on first execution. Tell the user before that download. A local model directory can instead be supplied with `--nn`.

The fork's README currently shows an install command for `paulgp/econ-ai-detector`; use the pinned repository above unless the user explicitly chooses the upstream source.

## Execution

Resolve the wrapper path relative to this skill's directory, then run it with the environment's Python:

```bash
<python> scripts/run_econ_detector.py manuscript.pdf
<python> scripts/run_econ_detector.py clean-prose.txt --no-gate
```

Supported operating points are `0.001`, `0.005`, and `0.01`; `0.001` is the default. Pass `--nn /path/to/model` for local weights.

The wrapper emits JSON to standard output. It adds window indices and previews to the detector's metrics so findings can be grounded without describing the LR probability as an authorship percentage.

## Scope limitations

- Calibration is for economics working-paper prose, not arbitrary genres.
- Score body prose rather than references, equations, or front matter.
- Fewer than two scored 250-word windows are insufficient for the paper-level rule.
- A flag is a diagnostic signal; a non-flag does not establish human authorship.
- Light editing and model drift can reduce sensitivity.
