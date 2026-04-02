#!/usr/bin/env python3
"""Constrained JSON generation using llguidance.

Generates JSON output from prompts using grammar-constrained decoding
to guarantee syntactically valid output.

Usage:
    # Baseline (no fine-tuning)
    python src/decode.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --data data/Restaurants_1_dev.jsonl --schema data/Restaurants_1_schema.json

    # With LoRA checkpoint
    python src/decode.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --checkpoint checkpoints/lora_epoch5 \
        --data data/Restaurants_1_dev.jsonl --schema data/Restaurants_1_schema.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from llguidance import LLInterpreter
    HAS_LLGUIDANCE = True
except ImportError:
    HAS_LLGUIDANCE = False


def load_model(model_name: str, checkpoint: str | None, device: str):
    """Load base model, optionally with a LoRA checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    if checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        print(f"Loaded LoRA checkpoint: {checkpoint}")

    model.eval()
    return model, tokenizer


def greedy_decode_unconstrained(model, tokenizer, prompt: str, device: str,
                                 max_new_tokens: int = 1500) -> str:
    """Simple greedy decoding without grammar constraints."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion
    generated_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def greedy_decode_constrained(model, tokenizer, prompt: str, schema: dict,
                               device: str, max_new_tokens: int = 1500) -> str:
    """Grammar-constrained greedy decoding using llguidance.

    Falls back to unconstrained if llguidance is not available.
    """
    if not HAS_LLGUIDANCE:
        print("Warning: llguidance not available, using unconstrained decoding",
              file=sys.stderr)
        return greedy_decode_unconstrained(model, tokenizer, prompt, device,
                                            max_new_tokens)

    # Build grammar from JSON schema
    schema_str = json.dumps(schema)
    grammar = f'{{"grammars": [{{"json_schema": {schema_str}}}]}}'

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    vocab_size = model.config.vocab_size

    interp = LLInterpreter(
        tokenizer_name=None,
        json_tokens=tokenizer.convert_ids_to_tokens(range(vocab_size)),
        tokenizer_eos_token=tokenizer.eos_token_id,
    )
    interp.start(grammar)

    generated_ids = list(input_ids)
    input_tensor = torch.tensor([generated_ids], device=device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            logits = outputs.logits[0, -1, :]

        # Get mask from llguidance
        mask = interp.compute_logit_bias()
        if mask is not None:
            logit_bias = torch.full((vocab_size,), float('-inf'), device=device)
            for token_id, bias in mask:
                logit_bias[token_id] = bias
            logits = logits + logit_bias

        next_token = logits.argmax().item()
        result = interp.advance_token(next_token)

        if result.stop:
            break

        generated_ids.append(next_token)
        input_tensor = torch.tensor([[next_token]], device=device)

    # Decode generated portion
    output_ids = generated_ids[len(input_ids):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Constrained JSON generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint path")
    parser.add_argument("--data", required=True, help="JSONL data file")
    parser.add_argument("--schema", required=True, help="JSON Schema file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--constrained", action="store_true", default=True,
                        help="Use grammar-constrained decoding (default)")
    parser.add_argument("--unconstrained", action="store_true",
                        help="Use unconstrained decoding")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    use_constrained = not args.unconstrained

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
    print(f"Constrained: {use_constrained}")
    print(f"Device: {args.device}")

    # Load model
    model, tokenizer = load_model(args.model, args.checkpoint, args.device)

    # Generate
    results = []
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]

        if use_constrained:
            output = greedy_decode_constrained(
                model, tokenizer, prompt, schema, args.device, args.max_tokens
            )
        else:
            output = greedy_decode_unconstrained(
                model, tokenizer, prompt, args.device, args.max_tokens
            )

        # Validate JSON
        valid_json = False
        try:
            parsed = json.loads(output)
            valid_json = True
        except json.JSONDecodeError:
            parsed = None

        results.append({
            "dialogue_id": ex.get("dialogue_id", f"example_{i}"),
            "generated_json": output,
            "valid_json": valid_json,
            "target_json": ex["target_json"],
        })

        status = "valid" if valid_json else "INVALID"
        print(f"  [{i+1}/{len(examples)}] {status} "
              f"({len(output)} chars)")

    # Write output
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    valid_count = sum(1 for r in results if r["valid_json"])
    print(f"\nValid JSON: {valid_count}/{len(results)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
