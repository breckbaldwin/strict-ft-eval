#!/bin/bash
# Download SGD dataset and prepare train/test splits for both schemas.
#
# Usage: bash scripts/prepare_data.sh

set -e

cd "$(dirname "$0")/.."

echo "=== Preparing data ==="

# 1. Download SGD dataset
if [ -d "data/sgd" ]; then
    echo "SGD dataset already exists, skipping download."
else
    echo "Downloading SGD dataset..."
    git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git data/sgd
fi

# 2. Extract examples for both schemas
echo ""
echo "--- Extracting Restaurants_1 ---"
python src/prepare_data.py --service Restaurants_1 --split train --max 300

echo ""
echo "--- Extracting Flights_1 ---"
python src/prepare_data.py --service Flights_1 --split train --max 300

# 3. Split into train/test (250/50)
echo ""
echo "--- Splitting train/test ---"
python3 -c "
import json, random
for service in ['Restaurants_1', 'Flights_1']:
    with open(f'data/{service}_train.jsonl') as f:
        examples = [json.loads(line) for line in f]
    random.seed(42)
    random.shuffle(examples)
    with open(f'data/{service}_train.jsonl', 'w') as f:
        for ex in examples[:250]:
            f.write(json.dumps(ex) + '\n')
    with open(f'data/{service}_test.jsonl', 'w') as f:
        for ex in examples[250:]:
            f.write(json.dumps(ex) + '\n')
    print(f'{service}: 250 train, {len(examples)-250} test')
"

# 4. Verify
echo ""
echo "=== Data files ==="
for f in data/Restaurants_1_train.jsonl data/Restaurants_1_test.jsonl \
         data/Flights_1_train.jsonl data/Flights_1_test.jsonl \
         data/Restaurants_1_schema.json data/Flights_1_schema.json; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $f ($lines lines)"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "Data preparation complete."
