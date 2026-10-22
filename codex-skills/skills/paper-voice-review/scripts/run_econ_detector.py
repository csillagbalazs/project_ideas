#!/usr/bin/env python3
"""Run econ-ai-detector with grounded, review-friendly JSON output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def _load_detector():
    try:
        from econ_ai_detector.detector import Detector
        from econ_ai_detector.preprocess import cut_references, pdf_text, prose_ok, windows
    except ImportError as exc:
        raise SystemExit(
            "econ_ai_detector is not installed in this Python environment. "
            "Read references/detector-setup.md and install the pinned detector "
            "in an isolated environment before retrying."
        ) from exc
    return Detector, cut_references, pdf_text, prose_ok, windows


def _preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("econ-ai-detector")
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score economics-paper prose and emit review-friendly JSON."
    )
    parser.add_argument("path", type=Path, help="PDF or UTF-8 text file")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="score every window; appropriate for already-clean prose",
    )
    parser.add_argument(
        "--target-fpr",
        default="0.001",
        choices=("0.001", "0.005", "0.01"),
        help="paper-level operating point",
    )
    parser.add_argument("--nn", default=None, help="local model directory or HF model ID")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=240,
        help="maximum normalized characters included for each window preview",
    )
    args = parser.parse_args()

    if args.preview_chars < 40:
        parser.error("--preview-chars must be at least 40")
    if not args.path.is_file():
        parser.error(f"input file does not exist: {args.path}")

    Detector, cut_references, pdf_text, prose_ok, make_windows = _load_detector()

    raw_bytes = args.path.read_bytes()
    if args.path.suffix.lower() == ".pdf":
        text = cut_references(pdf_text(str(args.path)))
        input_kind = "pdf"
    else:
        text = raw_bytes.decode("utf-8", errors="replace")
        input_kind = "text"

    all_windows = make_windows(text)
    if args.no_gate:
        selected = list(enumerate(all_windows))
    else:
        selected = [(index, value) for index, value in enumerate(all_windows) if prose_ok(value)]

    detector = Detector(nn_path=args.nn, target_fpr=args.target_fpr)
    kept_text = [value for _, value in selected]
    lr, nn, lr_k2, nn_k2, flagged = detector.score_windows(kept_text)
    eligible = len(kept_text) >= 2
    verdict = "flagged" if eligible and flagged else "not_flagged" if eligible else "insufficient_text"

    threshold_lr = float(detector.thr["lr_k2_min"])
    threshold_nn = float(detector.thr["nn_margin_k2_min"])
    window_rows = []
    for scored_index, ((source_index, value), lr_value, nn_value) in enumerate(
        zip(selected, lr.tolist(), nn.tolist())
    ):
        window_rows.append(
            {
                "scored_index": scored_index,
                "source_window_index": source_index,
                "word_count": len(value.split()),
                "lr_probability": float(lr_value),
                "nn_margin": float(nn_value),
                "above_lr_threshold": bool(lr_value >= threshold_lr),
                "above_nn_threshold": bool(nn_value >= threshold_nn),
                "above_both_thresholds": bool(
                    lr_value >= threshold_lr and nn_value >= threshold_nn
                ),
                "preview": _preview(value, args.preview_chars),
            }
        )

    output = {
        "tool": "econ-ai-detector",
        "package_version": _package_version(),
        "input": {
            "path": str(args.path.resolve()),
            "kind": input_kind,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "prose_gate": not args.no_gate,
        },
        "result": {
            "verdict": verdict,
            "paper_rule_eligible": eligible,
            "flagged": bool(flagged) if eligible else None,
            "target_fpr": args.target_fpr,
            "total_windows": len(all_windows),
            "scored_windows": len(kept_text),
            "lr_k2": float(lr_k2) if eligible else None,
            "nn_margin_k2": float(nn_k2) if eligible else None,
            "thresholds": {
                "lr_k2_min": threshold_lr,
                "nn_margin_k2_min": threshold_nn,
            },
        },
        "windows": window_rows,
        "interpretation": {
            "not_an_ai_percentage": True,
            "note": (
                "This is a diagnostic classifier calibrated on economics working-paper "
                "prose. It does not prove authorship; a non-flag is weak evidence."
            ),
        },
    }
    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
