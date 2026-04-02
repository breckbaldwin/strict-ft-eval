#!/usr/bin/env python3
"""Post-hoc per-grammar-role loss decomposition for structured JSON output.

Given a JSON string and its schema, assigns each character a grammar role,
maps roles to BPE token positions, and computes per-role loss from
teacher-forced log-probabilities.

Usage:
    # Compute per-role loss for a model on held-out data
    python src/decompose.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/Restaurants_1_dev.jsonl --schema data/Restaurants_1_schema.json

    # With a LoRA checkpoint
    python src/decompose.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --checkpoint checkpoints/restaurants_lora_epoch5 \
        --data data/Restaurants_1_dev.jsonl --schema data/Restaurants_1_schema.json
"""

import argparse
import json
import sys
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Grammar role assignment
# ---------------------------------------------------------------------------

class GrammarRole(Enum):
    STRUCTURAL = auto()   # { } [ ] : ,
    QUOTE = auto()        # "
    KEY = auto()          # object key characters
    ENUM_VALUE = auto()   # categorical value characters
    BOOLEAN = auto()      # True / False string characters
    NUMBER = auto()       # numeric characters
    FREE_TEXT = auto()    # non-categorical string value characters
    WHITESPACE = auto()   # spaces, tabs, newlines
    UNKNOWN = auto()      # fallback


def assign_grammar_roles(json_str: str, schema: dict) -> list[GrammarRole]:
    """Walk a JSON string character by character, assigning grammar roles.

    Returns a list of GrammarRole, one per character in json_str.
    """
    roles = [GrammarRole.UNKNOWN] * len(json_str)

    # Build lookup: key -> field info from schema
    properties = schema.get("properties", {})
    enum_fields = set()
    boolean_fields = set()
    enum_values_by_field = {}
    for key, prop in properties.items():
        if "enum" in prop:
            vals = set(prop["enum"])
            if vals == {"True", "False"}:
                boolean_fields.add(key)
            else:
                enum_fields.add(key)
                enum_values_by_field[key] = vals

    # Simple recursive-descent JSON walker
    pos = 0

    def skip_ws():
        nonlocal pos
        while pos < len(json_str) and json_str[pos] in ' \t\n\r':
            roles[pos] = GrammarRole.WHITESPACE
            pos += 1

    def parse_string() -> str:
        """Parse a JSON string, assigning QUOTE to quotes and returning content."""
        nonlocal pos
        assert json_str[pos] == '"'
        roles[pos] = GrammarRole.QUOTE
        pos += 1

        start = pos
        while pos < len(json_str):
            if json_str[pos] == '\\':
                pos += 2  # skip escape sequence
                continue
            if json_str[pos] == '"':
                content = json_str[start:pos]
                roles[pos] = GrammarRole.QUOTE
                pos += 1
                return content
            pos += 1
        return json_str[start:]

    def assign_string_content_roles(start: int, end: int, role: GrammarRole):
        """Assign a role to character positions [start, end)."""
        for i in range(start, end):
            if roles[i] == GrammarRole.UNKNOWN:
                roles[i] = role

    def parse_value(current_key: str | None = None):
        """Parse a JSON value, assigning roles based on context."""
        nonlocal pos
        skip_ws()
        if pos >= len(json_str):
            return

        ch = json_str[pos]

        if ch == '"':
            # String value — determine role from key context
            content_start = pos + 1  # after opening quote
            content = parse_string()
            content_end = pos - 1    # before closing quote

            # Determine role based on field type
            if current_key in boolean_fields:
                role = GrammarRole.BOOLEAN
            elif current_key in enum_fields:
                role = GrammarRole.ENUM_VALUE
            else:
                role = GrammarRole.FREE_TEXT

            assign_string_content_roles(content_start, content_end, role)

        elif ch == '{':
            parse_object()
        elif ch == '[':
            parse_array()
        elif ch in '-0123456789':
            parse_number()
        elif json_str[pos:pos+4] == 'true':
            for i in range(4):
                roles[pos] = GrammarRole.BOOLEAN
                pos += 1
        elif json_str[pos:pos+5] == 'false':
            for i in range(5):
                roles[pos] = GrammarRole.BOOLEAN
                pos += 1
        elif json_str[pos:pos+4] == 'null':
            for i in range(4):
                roles[pos] = GrammarRole.STRUCTURAL
                pos += 1

    def parse_number():
        nonlocal pos
        while pos < len(json_str) and json_str[pos] in '-0123456789.eE+':
            roles[pos] = GrammarRole.NUMBER
            pos += 1

    def parse_object():
        nonlocal pos
        assert json_str[pos] == '{'
        roles[pos] = GrammarRole.STRUCTURAL
        pos += 1

        skip_ws()
        if pos < len(json_str) and json_str[pos] == '}':
            roles[pos] = GrammarRole.STRUCTURAL
            pos += 1
            return

        while pos < len(json_str):
            skip_ws()
            # Key
            key_content_start = pos + 1
            key = parse_string()
            key_content_end = pos - 1
            assign_string_content_roles(key_content_start, key_content_end, GrammarRole.KEY)

            skip_ws()
            # Colon
            if pos < len(json_str) and json_str[pos] == ':':
                roles[pos] = GrammarRole.STRUCTURAL
                pos += 1

            skip_ws()
            # Value
            parse_value(current_key=key)

            skip_ws()
            if pos < len(json_str) and json_str[pos] == ',':
                roles[pos] = GrammarRole.STRUCTURAL
                pos += 1
            elif pos < len(json_str) and json_str[pos] == '}':
                roles[pos] = GrammarRole.STRUCTURAL
                pos += 1
                return
            else:
                break

    def parse_array():
        nonlocal pos
        assert json_str[pos] == '['
        roles[pos] = GrammarRole.STRUCTURAL
        pos += 1

        skip_ws()
        if pos < len(json_str) and json_str[pos] == ']':
            roles[pos] = GrammarRole.STRUCTURAL
            pos += 1
            return

        while pos < len(json_str):
            skip_ws()
            parse_value()
            skip_ws()
            if pos < len(json_str) and json_str[pos] == ',':
                roles[pos] = GrammarRole.STRUCTURAL
                pos += 1
            elif pos < len(json_str) and json_str[pos] == ']':
                roles[pos] = GrammarRole.STRUCTURAL
                pos += 1
                return
            else:
                break

    skip_ws()
    if pos < len(json_str):
        if json_str[pos] == '{':
            parse_object()
        elif json_str[pos] == '[':
            parse_array()
        else:
            parse_value()

    return roles


def map_roles_to_tokens(json_str: str, token_ids: list[int],
                        char_roles: list[GrammarRole],
                        tokenizer) -> list[GrammarRole]:
    """Map character-level grammar roles to token-level roles.

    For tokens spanning multiple grammar roles (cross-terminal tokens),
    assigns the role of the first character. This is a known limitation
    of post-hoc decomposition — noted in the paper.
    """
    # Decode each token to find its character span
    token_roles = []

    # Reconstruct character offsets by decoding token by token
    decoded_so_far = ""
    for tid in token_ids:
        token_text = tokenizer.decode([tid])
        start = len(decoded_so_far)
        end = start + len(token_text)
        decoded_so_far += token_text

        # Assign role of first character in the token's span
        if start < len(char_roles):
            token_roles.append(char_roles[start])
        else:
            token_roles.append(GrammarRole.UNKNOWN)

    return token_roles


# ---------------------------------------------------------------------------
# Teacher-forced loss computation
# ---------------------------------------------------------------------------

def compute_teacher_forced_loss(
    model, tokenizer, prompt: str, target_json: str, device: str
) -> tuple[list[int], list[float]]:
    """Compute per-token cross-entropy loss on target_json given prompt.

    Returns (token_ids, per_token_losses) for the target portion only.
    """
    # Tokenize prompt and target separately to know the boundary
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    target_ids = tokenizer.encode(target_json, add_special_tokens=False)

    input_ids = prompt_ids + target_ids
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Compute per-token loss on target portion only
    # logits[t] predicts token[t+1], so for target starting at position P:
    #   logits[P-1] predicts target[0], logits[P] predicts target[1], etc.
    prompt_len = len(prompt_ids)
    target_len = len(target_ids)

    # Shift: logits[prompt_len-1 : prompt_len-1+target_len] predict target_ids
    pred_logits = logits[0, prompt_len-1 : prompt_len-1+target_len, :]
    target_tensor = torch.tensor(target_ids, device=device)

    per_token_loss = F.cross_entropy(
        pred_logits, target_tensor, reduction='none'
    )

    return target_ids, per_token_loss.cpu().tolist()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_role(
    token_ids: list[int], losses: list[float], token_roles: list[GrammarRole]
) -> dict[str, dict]:
    """Aggregate loss by grammar role.

    Returns dict mapping role name to {mean_loss, total_loss, count, char_count}.
    """
    role_stats = defaultdict(lambda: {"total_loss": 0.0, "count": 0})

    for tid, loss, role in zip(token_ids, losses, token_roles):
        name = role.name
        role_stats[name]["total_loss"] += loss
        role_stats[name]["count"] += 1

    for name, stats in role_stats.items():
        stats["mean_loss"] = stats["total_loss"] / stats["count"] if stats["count"] > 0 else 0.0

    return dict(role_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(model_name: str, checkpoint: str | None, device: str):
    """Load base model, optionally with a LoRA checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True
    ).to(device)

    if checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        print(f"Loaded LoRA checkpoint: {checkpoint}")

    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Per-grammar-role loss decomposition"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint path")
    parser.add_argument("--data", required=True, help="JSONL data file")
    parser.add_argument("--schema", required=True, help="JSON Schema file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output TSV file")
    args = parser.parse_args()

    # Load schema
    with open(args.schema) as f:
        schema = json.load(f)

    # Load data
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'none (baseline)'}")
    print(f"Examples: {len(examples)}")
    print(f"Device: {args.device}")

    # Load model
    model, tokenizer = load_model(args.model, args.checkpoint, args.device)

    # Process examples
    all_role_stats = defaultdict(lambda: {"total_loss": 0.0, "count": 0})

    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        target_json = ex["target_json"]

        # Compute teacher-forced loss
        token_ids, losses = compute_teacher_forced_loss(
            model, tokenizer, prompt, target_json, args.device
        )

        # Assign grammar roles
        char_roles = assign_grammar_roles(target_json, schema)
        token_roles = map_roles_to_tokens(target_json, token_ids, char_roles, tokenizer)

        # Aggregate
        example_stats = aggregate_by_role(token_ids, losses, token_roles)

        # Accumulate across examples
        for role_name, stats in example_stats.items():
            all_role_stats[role_name]["total_loss"] += stats["total_loss"]
            all_role_stats[role_name]["count"] += stats["count"]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processed {i+1}/{len(examples)}")

    # Compute means
    print(f"\n{'='*60}")
    print(f"Per-Grammar-Role Loss Decomposition")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint or 'baseline (no fine-tuning)'}")
    print(f"Examples: {len(examples)}")
    print(f"{'='*60}")
    print(f"{'Role':<15} {'Mean Loss':>10} {'Tokens':>8}")
    print(f"{'-'*35}")

    total_loss = 0.0
    total_tokens = 0
    role_order = [
        "STRUCTURAL", "QUOTE", "KEY", "ENUM_VALUE", "BOOLEAN",
        "NUMBER", "FREE_TEXT", "WHITESPACE", "UNKNOWN"
    ]

    results = {}
    for role_name in role_order:
        if role_name not in all_role_stats:
            continue
        stats = all_role_stats[role_name]
        mean = stats["total_loss"] / stats["count"] if stats["count"] > 0 else 0.0
        print(f"{role_name:<15} {mean:>10.4f} {stats['count']:>8}")
        total_loss += stats["total_loss"]
        total_tokens += stats["count"]
        results[role_name] = {"mean_loss": mean, "count": stats["count"]}

    if total_tokens > 0:
        print(f"{'-'*35}")
        print(f"{'TOTAL':<15} {total_loss/total_tokens:>10.4f} {total_tokens:>8}")

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "checkpoint": args.checkpoint,
                "num_examples": len(examples),
                "per_role": results,
                "total_mean_loss": total_loss / total_tokens if total_tokens > 0 else 0.0,
                "total_tokens": total_tokens,
            }, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
