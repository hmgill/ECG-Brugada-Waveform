#!/usr/bin/env python3
"""
ECG inference CLI — predict Brugada syndrome and cardiac abnormalities.

Usage:
    # Single WFDB recording
    python predict.py \\
        --input recordings/patient_001 \\
        --checkpoint checkpoints/best.ckpt \\
        --config config/train.yaml \\
        --thresholds config/thresholds.json

    # Directory (scans recursively for .hea files)
    python predict.py \\
        --input recordings/ \\
        --checkpoint checkpoints/best.ckpt \\
        --config config/train.yaml \\
        --thresholds config/thresholds.json \\
        --output results/predictions.csv   # .csv for table, .json for json

    # CSV with a 'path' column
    python predict.py \\
        --input recordings.csv \\
        --checkpoint checkpoints/best.ckpt \\
        --config config/train.yaml \\
        --thresholds config/thresholds.json \\
        --output results/predictions.json \\
        --format json

Output formats:
    table   (default) prints a compact one-line-per-recording table to stdout
            and writes a CSV file if --output is specified
    summary verbose per-recording text block
    json    full probabilities for all classes; writes JSON if --output given
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure agent/ root is on the path when running from inference/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.engine import ECGInference, SUPERCLASS_NAMES


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_paths(input_path: Path) -> list[Path]:
    """Resolve --input to a flat list of WFDB recording paths."""
    if input_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(input_path)
        if "path" not in df.columns:
            raise ValueError("CSV must contain a 'path' column.")
        return [Path(p) for p in df["path"].dropna()]

    if input_path.is_dir():
        return sorted({p.with_suffix("") for p in input_path.rglob("*.hea")})

    # Single file — strip extension if provided
    return [Path(str(input_path).removesuffix(".hea").removesuffix(".dat"))]


def print_table(paths, results) -> None:
    """Print a compact summary table to stdout."""
    header = (
        f"{'Recording':<40} {'BRUG_p':>7} "
        f"{'Superclass':<20} {'Subclass'}"
    )
    print(header)
    print("-" * len(header))
    for path, result in zip(paths, results):
        if result is None:
            print(f"{str(path)[-40:]:<40}  ERROR")
            continue
        sup = ", ".join(result.active_superclasses) or "-"
        sub = ", ".join(result.active_subclasses) or "-"
        print(
            f"{str(path)[-40:]:<40} "
            f"{result.brugada_probability:>7.4f} "
            f"{sup:<20} {sub}"
        )


def write_csv(output_path: Path, paths, results) -> None:
    """Write full probability table as CSV."""
    # Derive subclass names from first successful result
    subclass_names = next(
        (r.subclass_names for r in results if r is not None), []
    )

    csv_path = output_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: metadata + per-class probabilities
        writer.writerow([
            "recording",
            "brugada_probability",
            "brugada_positive",
            "active_superclasses",
            "active_subclasses",
            *[f"sup_prob_{n}" for n in SUPERCLASS_NAMES],
            *[f"sub_prob_{n}" for n in subclass_names],
        ])
        for path, result in zip(paths, results):
            if result is None:
                writer.writerow([str(path), "ERROR"])
                continue
            writer.writerow([
                str(path),
                f"{result.brugada_probability:.6f}",
                int(result.brugada_positive),
                ";".join(result.active_superclasses),
                ";".join(result.active_subclasses),
                *[f"{p:.6f}" for p in result.sup_probs],
                *[f"{p:.6f}" for p in result.sub_probs],
            ])

    print(f"Results written to: {csv_path}")


def write_json(output_path: Path, paths, results) -> None:
    """Write full probability dict as JSON."""
    output = {
        str(path): (result.to_dict() if result is not None else {"error": True})
        for path, result in zip(paths, results)
    }
    json_path = output_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to: {json_path}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECG inference: Brugada syndrome and cardiac abnormality detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="WFDB recording path, directory of recordings, or CSV with 'path' column",
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True, type=Path,
        help="Path to .ckpt checkpoint file",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to train.yaml used for this checkpoint",
    )
    parser.add_argument(
        "--thresholds", "-t", required=True, type=Path,
        help="Path to thresholds.json from find_thresholds.py",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help=(
            "Output file path. Format determines extension: "
            "table → .csv, json → .json, summary → no file written unless specified."
        ),
    )
    parser.add_argument(
        "--format", choices=["table", "json", "summary"], default="table",
        help=(
            "Output format: "
            "'table' = compact one-line-per-recording printed to stdout, CSV saved if --output given; "
            "'json'  = full probabilities for all classes; "
            "'summary' = verbose per-recording text block"
        ),
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda', 'cpu', or None (auto)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Collect input paths ───────────────────────────────────────────────────
    paths = collect_paths(args.input)
    if not paths:
        print(f"No WFDB recordings found at: {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths)} recording(s).")

    # ── Load model ────────────────────────────────────────────────────────────
    engine = ECGInference.from_checkpoint(
        checkpoint=args.checkpoint,
        config=args.config,
        thresholds=args.thresholds,
        device=args.device,
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    results = engine.predict_batch(paths, verbose=True)

    n_ok  = sum(r is not None for r in results)
    n_pos = sum(r is not None and r.brugada_positive for r in results)
    print(f"\nProcessed {n_ok}/{len(paths)} recordings — "
          f"{n_pos} Brugada positive ({100*n_pos/max(n_ok,1):.1f}%)")

    # ── Display + save ────────────────────────────────────────────────────────
    if args.format == "table":
        print()
        print_table(paths, results)
        if args.output:
            write_csv(args.output, paths, results)

    elif args.format == "summary":
        for path, result in zip(paths, results):
            print(f"\n{'='*60}")
            print(f"Recording: {path}")
            print("=" * 60)
            if result is None:
                print("  ERROR: failed to load recording.")
            else:
                print(result.summary())
        if args.output:
            # Summary format saves JSON when --output is given
            write_json(args.output, paths, results)

    elif args.format == "json":
        output_dict = {
            str(path): (result.to_dict() if result is not None else {"error": True})
            for path, result in zip(paths, results)
        }
        if args.output:
            write_json(args.output, paths, results)
        else:
            print(json.dumps(output_dict, indent=2))


if __name__ == "__main__":
    main()
