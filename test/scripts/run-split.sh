#!/usr/bin/env bash

set -euox pipefail

TRAIN="hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/split"

TYPE=${1:-"random"}

HF_HOME="$CACHE" eluent split \
    "$TRAIN" \
    --train .7 \
    --validation .15 \
    -S smiles \
    --type "$TYPE" \
    -n 2 \
    --seed 42 \
    --cache "$CACHE" \
    --output "$OUTPUT"/$TYPE.csv \
    --plot "$OUTPUT"/$TYPE-plot.png
