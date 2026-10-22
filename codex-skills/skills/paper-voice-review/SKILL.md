---
name: paper-voice-review
description: Review an economics or social-science manuscript for paper readiness, AI-style writing signals, and consistency with the user's established author-voice profile. Use for diagnostic feedback on a draft or a second pass after revision. Do not use it to prove authorship, game a detector, or silently rewrite an entire paper.
---

# Paper Voice Review

Produce an evidence-backed editorial review that helps the user make the paper more specific, natural, and consistent with their own demonstrated voice. Preserve claims, citations, notation, and intended meaning. Treat detector output as one fallible diagnostic signal, never as the goal of revision.

## Inputs and scope

Accept a local PDF, TXT, Markdown, or LaTeX manuscript, or a grounded Google Drive document. Resolve the exact target before analysis. Unless the user asks for a full paper review, prioritize the abstract, introduction, main interpretation passages, and conclusion; detector calibration is strongest on ordinary economics-paper prose.

Use the canonical author profile at `${CODEX_HOME}/author-voice/profile.md`, or `~/.codex/author-voice/profile.md` when `CODEX_HOME` is unset. If absent, look for `.codex/author-voice/profile.md` in the project. If no profile exists or it is materially stale, use the `author-voice-profile` skill first. Do not substitute a generic style guide for missing personal evidence.

## Review sequence

1. Preserve the original manuscript. Work from a temporary clean-text extraction when preprocessing is needed.
2. Separate prose from bibliography, tables, equations, code, appendices, boilerplate, and long quotations.
3. For PDFs, use the detector's PDF extraction and prose gate. For already cleaned prose, pass a TXT file with `--no-gate`. Convert Markdown and LaTeX to clean prose before detection rather than scoring markup.
4. Run [scripts/run_econ_detector.py](scripts/run_econ_detector.py) with the isolated detector environment described in [references/detector-setup.md](references/detector-setup.md). Use the default `0.001` operating point unless the user specifies another supported point.
5. Apply the contextual checklist in [references/ai-writing-signs.md](references/ai-writing-signs.md). Inspect substance and citation support before surface phrasing.
6. Compare the draft with the author profile. Identify both conspicuous mismatches and passages that already match well.
7. Consolidate overlapping findings and rank them by editorial value. Follow [references/review-output-format.md](references/review-output-format.md).

## Detector interpretation

Report the paper verdict, eligibility of the paper-level rule, number of windows scored, thresholds, second-highest LR probability, second-highest neural margin, and the most diagnostic windows. Never convert these values into a percentage of text "written by AI."

The detector uses 250-word windows and requires at least two scored windows for its paper-level rule. For shorter inputs, say `insufficient text`; do not call the text clean. A flag is evidence of a learned stylistic pattern, not proof of provenance. A non-flag is weak evidence and must not override substantive review.

Do not repeatedly mutate prose merely to lower a score. After genuine editorial revisions, one comparison run is appropriate; report both improvements and regressions without promising a passing result.

## Feedback standard

For each material issue provide:

- Location and a short exact excerpt.
- Category and severity.
- Why it weakens clarity, specificity, evidentiary support, or voice consistency.
- Evidence from the author's profile when claiming a voice mismatch.
- A revision direction that preserves the intended claim.
- At most one conservative example rewrite when it materially clarifies the direction.

Prefer feedback over wholesale rewriting. If the user later asks for implementation, edit only the authorized artifact and keep a change log. Never fabricate citations or strengthen causal claims beyond the evidence.

## Integrity boundary

The aim is authentic editing in the user's established voice, not concealment of prohibited authorship. Preserve disclosure obligations imposed by the journal, institution, or collaborator. Do not claim that any detector can certify human authorship.
