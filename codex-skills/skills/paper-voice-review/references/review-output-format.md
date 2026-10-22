# Paper voice review format

Keep the report concise enough to act on. Combine duplicate symptoms that share one underlying issue.

## 1. Editorial assessment

In two or three sentences state the most important paper-readiness issue, the overall voice match, and the detector verdict with an explicit caveat. Do not declare the manuscript human- or AI-authored.

## 2. Detector diagnostics

Report:

- `verdict`: flagged, not flagged, or insufficient text
- operating point and thresholds
- scored and total windows
- second-highest LR probability and neural margin
- diagnostic windows, identified by index and short preview
- register or extraction limitations

Do not report an "AI percentage."

## 3. Voice comparison

Use a compact table with these columns:

| Dimension | Draft | Established profile | Assessment |
|---|---|---|---|

Include only supported dimensions. Distinguish a true mismatch from normal genre variation.

## 4. Priority revisions

Give at most ten consolidated priorities unless the user requests exhaustive comments. Order them:

1. Unsupported, inaccurate, or citation-sensitive claims.
2. Generic reasoning that obscures the paper's contribution or mechanism.
3. Conspicuous departures from the author's voice.
4. Repetitive stylistic signals.
5. Cosmetic issues.

For each priority provide location, excerpt, severity, rationale, profile evidence when relevant, and a revision direction. Use one example rewrite only when it makes the advice substantially clearer.

## 5. What already works

Identify two to five passages or habits that are specific, well-supported, and consistent with the user's voice. This prevents unnecessary homogenization.

## 6. Suggested revision sequence

End with a short order of operations. Recommend a detector rerun only after substantive edits, not after individual synonym substitutions.

## Severity labels

- `High`: factual, citation, causal, or argument-integrity problem.
- `Medium`: important clarity, specificity, or voice mismatch.
- `Low`: localized rhythm, diction, or formatting issue.
