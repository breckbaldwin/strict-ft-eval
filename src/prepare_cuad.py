#!/usr/bin/env python3
"""Prepare CUAD (Contract Understanding Atticus Dataset) data for the
per-grammar-role JSON extraction benchmark.

Produces train/test JSONL matching the existing Flights_1 / Restaurants_1
format (fields: `dialogue_id`, `prompt`, `target_json`) so downstream
decompose.py / decode.py / evaluate.py / margin_gating_eval.py all work
unchanged.

**Schema (8 boolean + 3 enum fields):** boolean presence flags span a
False-majority prevalence spectrum (5%–73%); enum fields add multi-valued
grammar-role diversity plus natural `not_specified` cases for evidential
gating.

Booleans (same False-majority shape as Flights_1 `refundable`):

| Field                       | CUAD clause              | Yes rate |
|-----------------------------|--------------------------|---------:|
| has_anti_assignment         | Anti-Assignment          |      73% |
| has_cap_on_liability        | Cap On Liability         |      54% |
| has_audit_rights            | Audit Rights             |      42% |
| has_exclusivity             | Exclusivity              |      35% |
| has_change_of_control       | Change Of Control        |      24% |
| has_non_compete             | Non-Compete              |      23% |
| has_liquidated_damages      | Liquidated Damages       |      12% |
| has_most_favored_nation     | Most Favored Nation      |       5% |

Enums (each includes `not_specified` — natural PCL underspecification):

| Field            | CUAD column         | Cardinality | `not_specified` rate |
|------------------|---------------------|------------:|---------------------:|
| governing_law    | Governing Law       |   12 values |       15% (76/510)   |
| renewal_term     | Renewal Term        |    6 values |       68% (347/510)  |
| expiration_type  | Expiration Date     |    3 values |       35% (181/510)  |

Usage:
    python src/prepare_cuad.py --split train --max 144
    python src/prepare_cuad.py --split test  --max 50

Reads:
    data/cuad/CUAD_v1/master_clauses.csv      — CUAD Yes/No labels (510 rows)
    data/cuad/CUAD_v1/full_contract_txt/*.txt — 200 contract texts (HF mirror)

Writes:
    data/cuad_train.jsonl
    data/cuad_test.jsonl
    data/cuad_schema.json
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple


# --- Boolean clause fields -----------------------------------------------

# short_field_name → CUAD "<Clause>" (matches the CSV column stem; add
# "-Answer" to get the Yes/No column).
BOOLEAN_FIELDS = {
    "has_anti_assignment":       "Anti-Assignment",
    "has_cap_on_liability":      "Cap On Liability",
    "has_audit_rights":          "Audit Rights",
    "has_exclusivity":           "Exclusivity",
    "has_change_of_control":     "Change Of Control",
    "has_non_compete":           "Non-Compete",
    "has_liquidated_damages":    "Liquidated Damages",
    "has_most_favored_nation":   "Most Favored Nation",
}


# --- Enum fields (normalized from free-text CUAD columns) ----------------

# Each entry: short_name → (csv_column, normalize_fn, enum_values_list)
# The normalize_fn takes the raw CSV cell and must return a value in the list.

GOVERNING_LAW_VALUES = [
    "New York", "California", "Delaware", "Texas", "Florida",
    "Pennsylvania", "People's Republic of China", "Nevada",
    "Ontario, Canada", "Illinois", "other", "not_specified",
]


def normalize_governing_law(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return "not_specified"
    lower = raw.lower()
    for v in GOVERNING_LAW_VALUES:
        if v in ("other", "not_specified"):
            continue
        if lower == v.lower():
            return v
    return "other"


RENEWAL_TERM_VALUES = [
    "successive_1_year", "successive_multi_year", "fixed_term",
    "perpetual", "other", "not_specified",
]

_RENEWAL_DURATION_RE = re.compile(r"\d+\s*(year|month|day|week)", re.IGNORECASE)


def normalize_renewal_term(raw: str) -> str:
    raw = raw.strip().lower()
    if not raw:
        return "not_specified"
    if "perpetual" in raw:
        return "perpetual"
    is_successive = "successive" in raw or "succesive" in raw  # tolerate CUAD typo
    if is_successive and re.search(r"\b1\s*year\b", raw):
        return "successive_1_year"
    if is_successive:
        return "successive_multi_year"
    if _RENEWAL_DURATION_RE.search(raw):
        return "fixed_term"
    return "other"


EXPIRATION_TYPE_VALUES = ["perpetual", "specific_date", "not_specified"]


def normalize_expiration_type(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return "not_specified"
    if raw.lower() == "perpetual":
        return "perpetual"
    return "specific_date"


ENUM_FIELDS: Dict[str, Tuple[str, Callable[[str], str], List[str]]] = {
    "governing_law":    ("Governing Law",   normalize_governing_law,   GOVERNING_LAW_VALUES),
    "renewal_term":     ("Renewal Term",    normalize_renewal_term,    RENEWAL_TERM_VALUES),
    "expiration_type":  ("Expiration Date", normalize_expiration_type, EXPIRATION_TYPE_VALUES),
}


# --- Combined schema -----------------------------------------------------

_SCHEMA_PROPS = {
    name: {"type": "string", "enum": ["True", "False"]}
    for name in BOOLEAN_FIELDS
}
for name, (_col, _fn, values) in ENUM_FIELDS.items():
    _SCHEMA_PROPS[name] = {"type": "string", "enum": values}

SCHEMA = {
    "type": "object",
    "title": "Contract clause presence + jurisdiction summary",
    "required": list(BOOLEAN_FIELDS) + list(ENUM_FIELDS),
    "additionalProperties": False,
    "properties": _SCHEMA_PROPS,
}


# --- Paths ----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CUAD_DIR = ROOT / "data" / "cuad" / "CUAD_v1"
CSV_PATH = CUAD_DIR / "master_clauses.csv"
TXT_ROOT = CUAD_DIR / "full_contract_txt"
OUT_DIR = ROOT / "data"


# --- Prompt template -------------------------------------------------------

PROMPT_TEMPLATE = """You are a JSON extraction system. Given the contract excerpt below, output a structured JSON object indicating which standard clauses are present in the contract.

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no code fences.
- Every field from the schema MUST be present, valued "True" or "False".
- "True" means the contract clearly contains that clause type. "False" means it does not.

Schema:
{schema}

Contract excerpt:
{contract}

JSON:"""


# --- Data loading ---------------------------------------------------------

def load_matched_rows(max_chars: int) -> List[Dict]:
    """Match CSV rows to txt files by filename stem. Return records with
    `id`, `text` (truncated to max_chars), and `labels` (8 True/False strings).
    """
    txts = {p.stem: p for p in TXT_ROOT.rglob("*.txt")}
    with open(CSV_PATH, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    records = []
    for row in csv_rows:
        stem = row["Filename"].rsplit(".", 1)[0]
        txt_path = txts.get(stem)
        if not txt_path:
            continue
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        text = text[:max_chars]
        labels = {}
        for short, clause in BOOLEAN_FIELDS.items():
            ans = row[f"{clause}-Answer"].strip().lower()
            labels[short] = "True" if ans == "yes" else "False"
        for short, (col, norm, _vals) in ENUM_FIELDS.items():
            labels[short] = norm(row[f"{col}-Answer"])
        records.append({
            "id": stem,
            "text": text,
            "labels": labels,
        })
    return records


def stratified_split(records: List[Dict], n_train: int, n_test: int,
                     seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split by total Yes-count bucket so train/test have similar label
    distributions. Deterministic given `seed`.
    """
    # Stratify by total "True" count across boolean fields — these are the
    # primary regression targets; enum fields are auxiliary.
    for r in records:
        r["_bucket"] = sum(
            1 for k in BOOLEAN_FIELDS if r["labels"].get(k) == "True"
        )
    rng = random.Random(seed)
    rng.shuffle(records)
    buckets: Dict[int, List[Dict]] = {}
    for r in records:
        buckets.setdefault(r["_bucket"], []).append(r)

    total_needed = n_train + n_test
    if len(records) < total_needed:
        raise SystemExit(
            f"Need {total_needed} records, only {len(records)} matched."
        )
    train, test = [], []
    for b, items in sorted(buckets.items()):
        p_train = n_train / total_needed
        n_b_train = round(len(items) * p_train)
        n_b_test = len(items) - n_b_train
        train.extend(items[:n_b_train])
        test.extend(items[n_b_train:n_b_train + n_b_test])
    # Trim to requested sizes if stratification over/under-shoots.
    train = train[:n_train]
    test = test[:n_test]
    for r in train + test:
        r.pop("_bucket", None)
    return train, test


# --- Rendering ------------------------------------------------------------

def build_prompt(text: str) -> str:
    schema_str = json.dumps({"properties": SCHEMA["properties"]}, indent=2)
    return PROMPT_TEMPLATE.format(schema=schema_str, contract=text)


def emit(records: List[Dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for r in records:
            target_json = json.dumps(r["labels"], indent=None)
            prompt = build_prompt(r["text"])
            record = {
                "dialogue_id": r["id"],
                "prompt": prompt,
                "target_json": target_json,
            }
            f.write(json.dumps(record) + "\n")


# --- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="Characters of contract text per example. Default 8000 "
                         "(~2000 tokens). Full contracts are ~40k chars.")
    ap.add_argument("--n-train", type=int, default=144)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to output filenames "
                         "(e.g. '_full' → cuad_train_full.jsonl). "
                         "Useful for materialising multiple truncation variants.")
    args = ap.parse_args()

    records = load_matched_rows(args.max_chars)
    print(f"Matched CSV rows to text files: {len(records)}")

    train, test = stratified_split(records, args.n_train, args.n_test, args.seed)
    print(f"Train: {len(train)}  Test: {len(test)}")

    # Schema file (always cuad_schema.json; identical regardless of truncation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = args.out_dir / "cuad_schema.json"
    with open(schema_path, "w") as f:
        json.dump(SCHEMA, f, indent=2)
    print(f"Wrote: {schema_path}")

    # Train/test JSONL — suffix lets us materialise multiple truncation variants
    suf = args.out_suffix
    train_path = args.out_dir / f"cuad_train{suf}.jsonl"
    test_path = args.out_dir / f"cuad_test{suf}.jsonl"
    emit(train, train_path)
    emit(test, test_path)
    print(f"Wrote: {train_path}  Wrote: {test_path}")

    # Label prevalence — booleans
    print("\nBoolean label prevalence (Yes rate):")
    for field in BOOLEAN_FIELDS:
        yes_train = sum(1 for r in train if r["labels"][field] == "True")
        yes_test = sum(1 for r in test if r["labels"][field] == "True")
        print(f"  {field:<30} train: {yes_train}/{len(train)} "
              f"({yes_train/len(train)*100:.0f}%)   test: {yes_test}/{len(test)} "
              f"({yes_test/len(test)*100:.0f}%)")

    # Label prevalence — enums
    from collections import Counter
    print("\nEnum label distribution (train + test combined):")
    for field in ENUM_FIELDS:
        dist = Counter(r["labels"][field] for r in train + test)
        total = sum(dist.values())
        print(f"  {field}:")
        for val, n in dist.most_common():
            print(f"    {val:<28} {n}/{total} ({n/total*100:.0f}%)")


if __name__ == "__main__":
    main()
