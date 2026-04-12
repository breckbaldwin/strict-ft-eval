#!/usr/bin/env python3
"""Presupposition labeling: relabel underspecified fields as 'ambiguous'.

For each constrained field, checks whether the input provides lexical
evidence for the field's value. If not, relabels the training target
as 'ambiguous' — the uniqueness presupposition is violated.

Usage:
    # Relabel Flights_1 data for the refundable field
    python src/presupposition_label.py \
        --data data/Flights_1_train.jsonl \
        --field refundable \
        --cue refund \
        --output data/Flights_1_train_pcl.jsonl

    # Also update the schema
    python src/presupposition_label.py \
        --schema data/Flights_1_schema.json \
        --field refundable \
        --output-schema data/Flights_1_schema_pcl.json
"""

import argparse
import json
import sys
from collections import Counter


def has_lexical_cue(prompt: str, cue: str) -> bool:
    """Check if the conversation portion of the prompt contains a lexical cue."""
    # Extract conversation from the prompt format
    conv_start = prompt.find('Conversation:\n')
    conv_end = prompt.find('\n\nJSON output:')
    if conv_start >= 0 and conv_end >= 0:
        conv = prompt[conv_start:conv_end]
    else:
        conv = prompt
    return cue.lower() in conv.lower()


def relabel_data(data_path: str, field: str, cue: str, output_path: str,
                 ambiguous_label: str = "ambiguous"):
    """Relabel a JSONL file: underspecified field values become 'ambiguous'."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))

    stats = Counter()
    output_records = []

    for rec in records:
        target = json.loads(rec['target_json'])
        prompt = rec['prompt']

        if field in target:
            if has_lexical_cue(prompt, cue):
                stats['kept'] += 1
            else:
                old_val = target[field]
                target[field] = ambiguous_label
                stats['relabeled'] += 1

        new_rec = dict(rec)
        new_rec['target_json'] = json.dumps(target, indent=2)
        output_records.append(new_rec)

    with open(output_path, 'w') as f:
        for rec in output_records:
            f.write(json.dumps(rec) + '\n')

    # Report
    print(f"Input:  {data_path} ({len(records)} examples)")
    print(f"Output: {output_path}")
    print(f"Field:  {field}, cue: '{cue}'")
    print(f"  Kept original label: {stats['kept']}")
    print(f"  Relabeled to '{ambiguous_label}': {stats['relabeled']}")

    # Verify distribution
    vals = [json.loads(r['target_json']).get(field) for r in output_records]
    print(f"  Distribution: {dict(Counter(vals))}")


def relabel_schema(schema_path: str, field: str, output_path: str,
                   ambiguous_label: str = "ambiguous"):
    """Add 'ambiguous' to a field's enum in the schema."""
    with open(schema_path) as f:
        schema = json.load(f)

    prop = schema.get('properties', {}).get(field, {})
    if 'enum' in prop:
        if ambiguous_label not in prop['enum']:
            prop['enum'].append(ambiguous_label)
            print(f"Added '{ambiguous_label}' to {field} enum: {prop['enum']}")
        else:
            print(f"'{ambiguous_label}' already in {field} enum")
    else:
        print(f"Warning: {field} has no enum in schema", file=sys.stderr)

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"Schema written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Presupposition labeling: relabel underspecified fields"
    )
    parser.add_argument("--data", help="Input JSONL data file")
    parser.add_argument("--schema", help="Input JSON Schema file")
    parser.add_argument("--field", required=True,
                        help="Field name to check/relabel")
    parser.add_argument("--cue", required=True,
                        help="Lexical cue to search for in conversation")
    parser.add_argument("--output", help="Output JSONL data file")
    parser.add_argument("--output-schema", help="Output JSON Schema file")
    parser.add_argument("--label", default="ambiguous",
                        help="Label for underspecified values (default: ambiguous)")
    args = parser.parse_args()

    if args.data and args.output:
        relabel_data(args.data, args.field, args.cue, args.output, args.label)

    if args.schema and args.output_schema:
        relabel_schema(args.schema, args.field, args.output_schema, args.label)

    if not args.data and not args.schema:
        parser.error("Provide --data and/or --schema")


if __name__ == "__main__":
    main()
