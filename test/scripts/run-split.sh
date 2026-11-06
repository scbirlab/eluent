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
    -k 2 \
    --seed 42 \
    --cache "$CACHE" \
    --output "$OUTPUT"/$TYPE.csv \
    --checkpoint "$OUTPUT"/$TYPE.el

HF_HOME="$CACHE" eluent write \
    "$OUTPUT"/$TYPE.el \
    --cache "$CACHE" \
    --output "$OUTPUT"/from-chk/$TYPE.csv

HF_HOME="$CACHE" eluent plot \
    "$OUTPUT"/$TYPE.el \
    --cache "$CACHE" \
    -S smiles \
    --seed 42 \
    --output "$OUTPUT"/from-chk/$TYPE-plot.png
