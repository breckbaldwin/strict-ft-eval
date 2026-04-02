# strict-ft-eval

Per-grammar-role loss decomposition for evaluating fine-tuned structured JSON output.

This repo accompanies the paper *"Valid JSON, Wrong Answer: Fine-Tuning Degrades Schema Key Prediction at Scale"*.

## Key Finding

Standard LoRA fine-tuning + grammar-constrained decoding produces valid JSON at all model scales. Aggregate loss metrics show clear improvement. But per-grammar-role decomposition reveals that **key prediction degrades** at 32B — the model memorizes training-set key ordering instead of learning the schema, while aggregate metrics hide the regression behind large gains on trivial structural tokens.

## Setup

```bash
git clone <repo-url>
cd strict-ft-eval
pip install -r requirements.txt
```

## Data Preparation

Uses the [Schema-Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) dataset (Rastogi et al., AAAI 2020, CC BY-SA 4.0).

```bash
# Download SGD dataset
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git data/sgd

# Extract Restaurants_1 (11 keys, 3 enum, 2 boolean, 6 free text)
python src/prepare_data.py --service Restaurants_1 --split train --max 300

# Extract Flights_1 (16 keys, 4 enum, 1 boolean, 11 free text)
python src/prepare_data.py --service Flights_1 --split train --max 300
```

## Local Usage

```bash
# Per-grammar-role decomposition on a baseline model
python src/decompose.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --data data/Restaurants_1_test.jsonl \
    --schema data/Restaurants_1_schema.json \
    --device cpu

# Standard LoRA fine-tuning
python src/train.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --data data/Restaurants_1_train.jsonl \
    --epochs 5 --device cpu

# Decompose the fine-tuned model
python src/decompose.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --checkpoint checkpoints/lora_epoch5 \
    --data data/Restaurants_1_test.jsonl \
    --schema data/Restaurants_1_schema.json \
    --device cpu
```

## RunPod (GPU Experiments)

The paper's results require GPU for 7B and 32B models. We use [RunPod](https://www.runpod.io/) for GPU instances.

### Hardware Requirements

| Scale | GPU | VRAM | Approx Cost |
|-------|-----|------|-------------|
| 0.5B  | Any | 8GB+ | — |
| 7B    | A40 | 48GB | ~$0.39/hr |
| 32B   | A100 80GB | 80GB | ~$1.19/hr |

### Launch and Setup

```bash
# Launch a pod (from your local machine)
python scripts/runpod_cloud.py launch --gpu "A100 PCIe" --name strict-ft-eval

# SSH into the pod
python scripts/runpod_cloud.py ssh

# On the pod: run setup (provide your repo URL and HuggingFace token)
bash scripts/setup_runpod.sh <GIT_REPO_URL> <HF_TOKEN>
```

### Running Experiments

```bash
# Smoke test (check GPU, libraries, data, model cache)
bash scripts/smoketest.sh

# Run all scales (32B first to fail fast on memory issues)
bash scripts/run_experiment.sh all

# Or run one scale at a time
bash scripts/run_experiment.sh 32b
bash scripts/run_experiment.sh 7b
bash scripts/run_experiment.sh 05b

# Summarize results into comparison tables
python scripts/summarize_results.py
```

Each scale runs: baseline decomposition → LoRA training (10 epochs) → fine-tuned decomposition, on both Restaurants_1 and Flights_1 schemas.

## Repository Structure

```
src/
  prepare_data.py    — Extract SGD dialogue→JSON pairs, build JSON schemas
  train.py           — Standard LoRA fine-tuning (PEFT, no custom adapters)
  decode.py          — Constrained JSON generation (llguidance)
  decompose.py       — Per-grammar-role loss decomposition (post-hoc)
  evaluate.py        — Evaluation: exact match, ROUGE-L, key coverage
scripts/
  smoketest.sh       — RunPod environment verification
  run_experiment.sh  — Run all experiments (decomposable by scale)
  setup_runpod.sh    — One-shot pod setup
  summarize_results.py — Aggregate results into paper tables
  runpod_cloud.py    — RunPod pod management (launch, ssh, stop, etc.)
data/
  sgd/               — Schema-Guided Dialogue dataset
  *_train.jsonl      — Training data (250 examples per schema)
  *_test.jsonl       — Test data (50 examples per schema)
  *_schema.json      — JSON Schema for constrained decoding
```

## Grammar Roles

The decomposition assigns each token in the generated JSON to one of:

| Role | Description | Examples |
|------|-------------|----------|
| STRUCTURAL | JSON syntax | `{` `}` `[` `]` `:` `,` |
| QUOTE | String delimiters | `"` |
| KEY | Object key characters | `city`, `cuisine`, `price_range` |
| ENUM_VALUE | Categorical values | `moderate`, `Italian`, `Economy` |
| BOOLEAN | Boolean strings | `True`, `False` |
| NUMBER | Numeric characters | `364`, `2` |
| FREE_TEXT | Non-categorical content | restaurant names, addresses |
| WHITESPACE | Formatting | spaces, newlines |

## Citation

```bibtex
@article{baldwin2026validjson,
  title={Valid JSON, Wrong Answer: Fine-Tuning Degrades Schema Key Prediction at Scale},
  author={Baldwin, Breck},
  year={2026},
  note={arXiv preprint}
}
```

## License

Code: MIT. Data: SGD dataset is CC BY-SA 4.0 (Google Research).
