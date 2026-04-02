#!/usr/bin/env python3
"""Extract dialogue-context → service-result JSON pairs from SGD.

Usage:
    python src/prepare_data.py --service Restaurants_1 --split train --max 200
    python src/prepare_data.py --service Flights_1 --split train --max 200
    python src/prepare_data.py --service Restaurants_1 --split dev --max 50

Outputs JSONL to data/{service}_{split}.jsonl with fields:
    {"dialogue_id": ..., "context": ..., "target_json": ..., "schema": ...}
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_schema(sgd_dir: Path, split: str, service_name: str) -> dict:
    """Load schema definition for a service from the split's schema.json."""
    schema_file = sgd_dir / split / "schema.json"
    with open(schema_file) as f:
        schemas = json.load(f)
    for s in schemas:
        if s["service_name"] == service_name:
            return s
    raise ValueError(f"Service {service_name} not found in {schema_file}")


def build_json_schema(service_schema: dict, include_soft_enums: bool = True) -> dict:
    """Convert SGD service schema to a JSON Schema for the service_result object.

    If include_soft_enums is True, slots with possible_values listed but
    is_categorical=False (like cuisine in Restaurants_1) are also treated
    as enums. This gives better grammar-role diversity for analysis.
    """
    properties = {}
    for slot in service_schema["slots"]:
        name = slot["name"]
        vals = slot.get("possible_values", [])
        is_cat = slot["is_categorical"]

        if is_cat and vals:
            if set(vals) == {"True", "False"}:
                properties[name] = {"type": "string", "enum": ["True", "False"]}
            else:
                properties[name] = {"type": "string", "enum": vals}
        elif include_soft_enums and vals:
            # Soft enum: not declared categorical but has listed values
            properties[name] = {"type": "string", "enum": vals}
        else:
            properties[name] = {"type": "string"}

    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


def extract_pairs(sgd_dir: Path, split: str, service_name: str,
                  max_examples: int) -> list[dict]:
    """Extract (dialogue_context, service_result) pairs."""
    dialogue_dir = sgd_dir / split
    pairs = []

    dialogue_files = sorted(dialogue_dir.glob("dialogues_*.json"))
    for dfile in dialogue_files:
        if len(pairs) >= max_examples:
            break

        with open(dfile) as f:
            dialogues = json.load(f)

        for dialogue in dialogues:
            if len(pairs) >= max_examples:
                break

            # Only process dialogues that use our target service
            if service_name not in dialogue.get("services", []):
                continue

            # Walk turns, accumulate context, extract service_results
            context_parts = []
            for turn in dialogue["turns"]:
                speaker = turn["speaker"]
                utterance = turn["utterance"]

                if speaker == "SYSTEM":
                    # Check for service_results from our target service
                    for frame in turn.get("frames", []):
                        if frame.get("service") != service_name:
                            continue
                        results = frame.get("service_results", [])
                        if not results:
                            continue

                        # We have a service result — use accumulated context
                        # and the first result as the target JSON
                        service_call = frame.get("service_call", {})
                        method = service_call.get("method", "unknown")

                        # Build the context string
                        context = "\n".join(context_parts)
                        if not context.strip():
                            continue

                        # Use first result as target
                        target = results[0]

                        pairs.append({
                            "dialogue_id": dialogue["dialogue_id"],
                            "method": method,
                            "context": context,
                            "target_json": json.dumps(target, indent=2),
                        })

                # Add this turn to context for future extractions
                label = "User" if speaker == "USER" else "Assistant"
                context_parts.append(f"{label}: {utterance}")

    return pairs


def make_prompt(context: str, json_schema: dict) -> str:
    """Create the instruction prompt for the model."""
    schema_str = json.dumps(json_schema, indent=2)
    return (
        f"Extract the structured information from the following conversation "
        f"into a JSON object conforming to this schema:\n\n"
        f"Schema:\n{schema_str}\n\n"
        f"Conversation:\n{context}\n\n"
        f"JSON output:"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare SGD data for fine-tuning")
    parser.add_argument("--service", required=True, help="SGD service name")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--max", type=int, default=200, help="Max examples to extract")
    parser.add_argument("--sgd-dir", default="data/sgd", help="Path to SGD dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    sgd_dir = Path(args.sgd_dir)
    output_dir = Path(args.output_dir)

    # Load schema
    service_schema = load_schema(sgd_dir, args.split, args.service)
    json_schema = build_json_schema(service_schema)

    print(f"Service: {args.service}")
    print(f"Split: {args.split}")
    print(f"Schema slots: {len(service_schema['slots'])}")
    categorical = [s for s in service_schema["slots"] if s["is_categorical"]]
    print(f"  Categorical: {len(categorical)}")
    print(f"  Free text: {len(service_schema['slots']) - len(categorical)}")

    # Extract pairs
    pairs = extract_pairs(sgd_dir, args.split, args.service, args.max)
    print(f"Extracted {len(pairs)} examples")

    if not pairs:
        print("No examples found!", file=sys.stderr)
        sys.exit(1)

    # Write output
    output_file = output_dir / f"{args.service}_{args.split}.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        for pair in pairs:
            record = {
                "dialogue_id": pair["dialogue_id"],
                "method": pair["method"],
                "prompt": make_prompt(pair["context"], json_schema),
                "target_json": pair["target_json"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {output_file}")

    # Also write the JSON schema for constrained decoding
    schema_file = output_dir / f"{args.service}_schema.json"
    with open(schema_file, "w") as f:
        json.dump(json_schema, f, indent=2)
    print(f"Wrote {schema_file}")

    # Print sample
    print(f"\n--- Sample (first example) ---")
    sample = pairs[0]
    print(f"Dialogue: {sample['dialogue_id']}")
    print(f"Method: {sample['method']}")
    print(f"Context (first 200 chars): {sample['context'][:200]}...")
    print(f"Target JSON:\n{sample['target_json']}")


if __name__ == "__main__":
    main()
