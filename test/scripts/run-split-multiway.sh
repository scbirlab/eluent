#!/usr/bin/env bash
set -euox pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

python - "$work_dir/input.csv" <<'PY'
import csv
import sys
from itertools import product

with open(sys.argv[1], "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["id", "scaffold", "taxon", "target"])
    for row_id, groups in enumerate(product(range(10), repeat=3)):
        writer.writerow([row_id, *groups])
PY

for folds in 1 3; do
    eluent split "$work_dir/input.csv" \
        --group-columns scaffold taxon target \
        --train 0.8 \
        --test 0.2 \
        --kfolds "$folds" \
        --seed 42 \
        --cache "$work_dir/cache" \
        --output "$work_dir/k${folds}/split.csv"
done
