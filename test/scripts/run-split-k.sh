#!/usr/bin/env bash

set -euox pipefail

TYPE=${1:-"random"}

TRAIN="hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/fold-split
mkdir -p "$OUTPUT_DIR"
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/split"


HF_HOME="$CACHE" eluent split \
    "$TRAIN" \
    --train .7 \
    --validation .15 \
    -S smiles \
    --type "$TYPE" \
    -K 5 \
    --seed 42 \
    --cache "$CACHE" \
    --output "$OUTPUT"/$TYPE.csv \
    --plot "$OUTPUT"/$TYPE-plot.png
