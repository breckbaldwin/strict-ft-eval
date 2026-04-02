#!/usr/bin/env python3
"""Summarize experiment results into comparison tables for the paper.

Usage: python scripts/summarize_results.py
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
SCALES = ["05b", "7b", "32b"]
DATASETS = ["restaurants", "flights"]
ROLE_ORDER = ["STRUCTURAL", "QUOTE", "KEY", "ENUM_VALUE", "BOOLEAN", "NUMBER", "FREE_TEXT", "WHITESPACE"]


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_comparison_table(dataset: str):
    """Print baseline vs fine-tuned comparison across scales."""
    print(f"\n{'='*80}")
    print(f"  {dataset.upper()} — Per-Grammar-Role Loss: Baseline vs Fine-Tuned")
    print(f"{'='*80}")

    # Header
    header = f"{'Role':<15}"
    for scale in SCALES:
        header += f" | {'Base':>8} {'FT':>8} {'Chg':>7}"
    print(header)
    print("-" * len(header))

    for role in ROLE_ORDER:
        row = f"{role:<15}"
        for scale in SCALES:
            baseline = load_result(RESULTS_DIR / f"{scale}_baseline_{dataset}.json")
            finetuned = load_result(RESULTS_DIR / f"{scale}_finetuned_{dataset}.json")

            if baseline and role in baseline.get("per_role", {}):
                base_loss = baseline["per_role"][role]["mean_loss"]
                row += f" | {base_loss:>8.4f}"
            else:
                row += f" | {'—':>8}"

            if finetuned and role in finetuned.get("per_role", {}):
                ft_loss = finetuned["per_role"][role]["mean_loss"]
                row += f" {ft_loss:>8.4f}"
            else:
                row += f" {'—':>8}"

            if (baseline and finetuned and
                role in baseline.get("per_role", {}) and
                role in finetuned.get("per_role", {})):
                base_loss = baseline["per_role"][role]["mean_loss"]
                ft_loss = finetuned["per_role"][role]["mean_loss"]
                if base_loss > 0:
                    pct = (ft_loss - base_loss) / base_loss * 100
                    marker = "!!" if pct > 0 else ""
                    row += f" {pct:>+6.1f}%{marker}"
                else:
                    row += f" {'—':>7}"
            else:
                row += f" {'—':>7}"

        print(row)

    # Total row
    row = f"{'TOTAL':<15}"
    for scale in SCALES:
        baseline = load_result(RESULTS_DIR / f"{scale}_baseline_{dataset}.json")
        finetuned = load_result(RESULTS_DIR / f"{scale}_finetuned_{dataset}.json")

        base_total = baseline.get("total_mean_loss", 0) if baseline else 0
        ft_total = finetuned.get("total_mean_loss", 0) if finetuned else 0

        row += f" | {base_total:>8.4f} {ft_total:>8.4f}"
        if base_total > 0 and ft_total > 0:
            pct = (ft_total - base_total) / base_total * 100
            row += f" {pct:>+6.1f}%"
        else:
            row += f" {'—':>7}"

    print("-" * len(header))
    print(row)


def print_key_regression_summary():
    """Highlight the KEY regression finding across scales."""
    print(f"\n{'='*80}")
    print(f"  KEY REGRESSION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scale':<8} {'Dataset':<15} {'Base KEY':>10} {'FT KEY':>10} {'Change':>10} {'Regressed?'}")
    print("-" * 65)

    for scale in SCALES:
        for dataset in DATASETS:
            baseline = load_result(RESULTS_DIR / f"{scale}_baseline_{dataset}.json")
            finetuned = load_result(RESULTS_DIR / f"{scale}_finetuned_{dataset}.json")

            base_key = "—"
            ft_key = "—"
            change = "—"
            regressed = ""

            if baseline and "KEY" in baseline.get("per_role", {}):
                base_key = f"{baseline['per_role']['KEY']['mean_loss']:.4f}"
            if finetuned and "KEY" in finetuned.get("per_role", {}):
                ft_key = f"{finetuned['per_role']['KEY']['mean_loss']:.4f}"

            if base_key != "—" and ft_key != "—":
                b = float(base_key)
                f_ = float(ft_key)
                pct = (f_ - b) / b * 100 if b > 0 else 0
                change = f"{pct:+.1f}%"
                if f_ > b:
                    regressed = "YES — REGRESSION"

            print(f"{scale:<8} {dataset:<15} {base_key:>10} {ft_key:>10} {change:>10} {regressed}")


def main():
    found = list(RESULTS_DIR.glob("*.json"))
    if not found:
        print("No results found in results/. Run experiments first.")
        sys.exit(1)

    print(f"Found {len(found)} result files.")

    for dataset in DATASETS:
        print_comparison_table(dataset)

    print_key_regression_summary()


if __name__ == "__main__":
    main()
