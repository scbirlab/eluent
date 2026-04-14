#!/usr/bin/env bash

set -euox pipefail

TRAIN="hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/split"

HF_HOME="$CACHE" eluent percentiles \
    "$TRAIN" \
    --columns clogp tpsa \
    --percentiles 1 5 10 \
    --batch 128 \
    --cache "$CACHE" \
    --output "$OUTPUT"/percentiles.csv \
    --plot "$OUTPUT"/percentiles-plot.png \
    --structure smiles

for type in faiss scaffold
do
    HF_HOME="$CACHE" eluent split \
        "$TRAIN" \
        --train .7 \
        --validation .15 \
        -S smiles \
        --type "$type" \
        -k 2 \
        --seed 42 \
        --cache "$CACHE" \
        --output "$OUTPUT"/$type.csv \
        --plot "$OUTPUT"/$type-plot.png
done