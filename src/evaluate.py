#!/usr/bin/env python3
"""Evaluate generated JSON against gold targets.

Metrics: exact match (enums, booleans), key coverage, ROUGE-L (free text).

Usage:
    python src/evaluate.py --predictions results/baseline.jsonl \
        --schema data/Restaurants_1_schema.json
"""

import argparse
import json
import sys
from collections import defaultdict


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between two strings (word-level)."""
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()

    if not pred_words or not ref_words:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == ref_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_pair(predicted: dict, target: dict, schema: dict) -> dict:
    """Evaluate a single predicted JSON object against the target."""
    properties = schema.get("properties", {})

    # Identify field types
    enum_fields = set()
    boolean_fields = set()
    free_text_fields = set()

    for key, prop in properties.items():
        if "enum" in prop:
            vals = set(prop["enum"])
            if vals == {"True", "False"}:
                boolean_fields.add(key)
            else:
                enum_fields.add(key)
        else:
            free_text_fields.add(key)

    results = {
        "key_coverage": 0.0,
        "enum_exact": [],
        "boolean_exact": [],
        "free_text_rouge": [],
        "predicted_keys": set(predicted.keys()),
        "target_keys": set(target.keys()),
    }

    # Key coverage
    target_keys = set(target.keys())
    predicted_keys = set(predicted.keys())
    if target_keys:
        results["key_coverage"] = len(predicted_keys & target_keys) / len(target_keys)

    # Per-field evaluation
    for key in target_keys:
        target_val = str(target.get(key, ""))
        pred_val = str(predicted.get(key, ""))

        if key in enum_fields:
            results["enum_exact"].append(1.0 if pred_val == target_val else 0.0)
        elif key in boolean_fields:
            results["boolean_exact"].append(1.0 if pred_val == target_val else 0.0)
        elif key in free_text_fields:
            results["free_text_rouge"].append(rouge_l_f1(pred_val, target_val))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated JSON")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL")
    parser.add_argument("--schema", required=True, help="JSON Schema file")
    args = parser.parse_args()

    with open(args.schema) as f:
        schema = json.load(f)

    results = []
    with open(args.predictions) as f:
        for line in f:
            results.append(json.loads(line))

    valid_count = 0
    key_coverages = []
    all_enum_exact = []
    all_boolean_exact = []
    all_free_text_rouge = []

    for r in results:
        if not r["valid_json"]:
            continue

        valid_count += 1
        try:
            predicted = json.loads(r["generated_json"])
            target = json.loads(r["target_json"])
        except json.JSONDecodeError:
            continue

        eval_result = evaluate_pair(predicted, target, schema)
        key_coverages.append(eval_result["key_coverage"])
        all_enum_exact.extend(eval_result["enum_exact"])
        all_boolean_exact.extend(eval_result["boolean_exact"])
        all_free_text_rouge.extend(eval_result["free_text_rouge"])

    print(f"Results: {len(results)} total, {valid_count} valid JSON")
    print(f"\nKey Coverage:     {sum(key_coverages)/len(key_coverages):.3f}" if key_coverages else "")
    print(f"Enum Exact:       {sum(all_enum_exact)/len(all_enum_exact):.3f}" if all_enum_exact else "")
    print(f"Boolean Exact:    {sum(all_boolean_exact)/len(all_boolean_exact):.3f}" if all_boolean_exact else "")
    print(f"Free Text ROUGE:  {sum(all_free_text_rouge)/len(all_free_text_rouge):.3f}" if all_free_text_rouge else "")

    # Overall score
    scores = []
    if all_enum_exact:
        scores.append(sum(all_enum_exact) / len(all_enum_exact))
    if all_boolean_exact:
        scores.append(sum(all_boolean_exact) / len(all_boolean_exact))
    if all_free_text_rouge:
        scores.append(sum(all_free_text_rouge) / len(all_free_text_rouge))
    if key_coverages:
        scores.append(sum(key_coverages) / len(key_coverages))

    if scores:
        overall = sum(scores) / len(scores)
        print(f"\nOverall Score:    {overall:.3f}")


if __name__ == "__main__":
    main()
