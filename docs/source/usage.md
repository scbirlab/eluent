# Usage

## Input data formats

**eluent** accepts two kinds of input:

- **Local files** — CSV (`.csv`), Parquet (`.parquet`), Arrow (`.arrow`), or a
  Hugging Face Datasets directory. The format is inferred automatically from the file
  extension.
- **Remote datasets** — any dataset hosted on the
  [🤗 Hugging Face Datasets Hub](https://huggingface.co/datasets), specified using a
  `hf://` URL:

```
hf://datasets/<owner>/<repo>~<config>:<split>
```

For example, to use the training split of the Fang 2023 ADME benchmark:

```
hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train
```

Output files are written in the format inferred from the `--output` filename extension.

---

## Command-line interface

Run `eluent --help` to see all sub-commands:

```
usage: eluent [-h] [--version] {split,percentiles} ...

Chemistry-aware splitting of large datasets.

options:
  -h, --help            show this help message and exit
  --version, -v         show program's version number and exit

Sub-commands:
  {split,percentiles}
    split               Make chemical train-test-val splits on out-of-core datasets.
    percentiles         Add columns indicating whether rows are in a percentile.
```

---

### `eluent split`

Partition a dataset into train / validation / test splits. The pipeline has two stages:

1. **Grouping** — each row is assigned to a group according to the chosen method.
2. **Bin-packing** — groups are packed into the requested split fractions. When `--seed`
   is provided, groups are packed randomly (reproducibly); otherwise a deterministic
   first-fit-decreasing algorithm is used.

#### Full option reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `input_file` | — | — | Input file or `hf://` URL. **Required.** |
| `--type` | — | `scaffold` | Grouping method: `random`, `scaffold`, or `faiss`. |
| `--structure` | `-S` | — | Column containing chemical structure strings. |
| `--input-representation` | `-R` | `smiles` | Structure format: `smiles`, `selfies`, `inchi`, `aa_seq`. |
| `--train` | — | — | Fraction of examples for training. **Required.** |
| `--validation` | — | inferred | Fraction of examples for validation. |
| `--test` | — | inferred | Fraction of examples for test. |
| `--kfolds` | `-K` | `1` | Generate _k_ cross-validation folds (overrides `--validation`). |
| `--seed` | `-i` | — | Random seed. Omit for deterministic packing. |
| `--n-neighbors` | `-k` | `10` | Number of nearest neighbours for FAISS grouping. |
| `--batch` | `-b` | `16` | Batch size for dataset mapping. |
| `--cache` | — | `.` | Directory for intermediate FAISS caches. |
| `--output` | `-o` | — | Output filename. **Required.** |
| `--plot` | — | — | Filename for UMAP embedding plot (requires `--structure`). |
| `--plot-sample` | `-n` | `20000` | Number of molecules to subsample for UMAP. |
| `--plot-seed` | `-e` | `42` | Random seed for UMAP layout. |
| `--extras` | `-x` | — | Additional columns to colour in the UMAP plot. |
| `--start` | — | `0` | First row of the dataset to process. |
| `--end` | — | end | Last row of the dataset to process. |

#### Splitting methods

##### Random

Each row is assigned a reproducible pseudo-random group ID by hashing its index with the
provided seed. Groups are then packed into splits.

```bash
$ eluent split \
    hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train \
    --type random \
    --structure smiles \
    --train 0.7 \
    --validation 0.15 \
    --seed 42 \
    --output split/random.csv \
    --plot split/random-plot.png
```

##### Scaffold

Molecules are grouped by their Murcko scaffold. All molecules sharing the same scaffold are
placed in the same split, preventing the model from seeing related scaffolds during both
training and evaluation. This is the default and the standard method for chemistry ML benchmarks.

```bash
$ eluent split \
    hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train \
    --type scaffold \
    --structure smiles \
    --train 0.7 \
    --validation 0.15 \
    --output split/scaffold.csv \
    --plot split/scaffold-plot.png
```

##### FAISS spectral splitting

Morgan fingerprints (2048-bit) are computed for each molecule. A binary k-NN graph is built
using a FAISS flat Hamming index. Connected components of the graph form groups, so all
molecules within a component are mutually similar.

```bash
$ eluent split \
    hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train \
    --type faiss \
    --structure smiles \
    --n-neighbors 10 \
    --train 0.7 \
    --validation 0.15 \
    --seed 42 \
    --cache ./cache \
    --output split/faiss.csv \
    --plot split/faiss-plot.png
```

#### k-fold cross-validation

Pass `--kfolds K` to generate _K_ train/validation splits. The `--validation` fraction is
distributed evenly among folds. Each fold is written to a separate subdirectory:

```
split/
  fold_0/
    scaffold_train.csv
    scaffold_validation.csv
  fold_1/
    scaffold_train.csv
    scaffold_validation.csv
  ...
```

```bash
$ eluent split \
    hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train \
    --type scaffold \
    --structure smiles \
    --train 0.7 \
    --validation 0.15 \
    --kfolds 5 \
    --seed 42 \
    --output split/scaffold.csv
```

---

### `eluent percentiles`

Add boolean annotation columns to a dataset indicating which rows fall within a top-_k_
percentile. Uses a T-Digest quantile sketch so the dataset never needs to be loaded into
memory in full.

For each `(column, percentile)` pair, a column named `<column>_top_<percentile>_pc` is
added, containing `True` for rows in the top percentile and `False` otherwise.

#### Full option reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `input_file` | — | — | Input file or `hf://` URL. **Required.** |
| `--columns` | `-c` | — | Column names to annotate. **Required.** |
| `--percentiles` | `-p` | `5` | Percentile thresholds (e.g. `1 5 10`). |
| `--reverse` | `-r` | — | Tag bottom percentiles instead of top. |
| `--compression` | `-z` | `500` | T-Digest centroids; higher is more accurate. |
| `--delta` | `-d` | `1.0` | Buffer width around the percentile cutoff for the "maybe" zone. |
| `--batch` | `-b` | `16` | Batch size. |
| `--cache` | — | `.` | Cache directory. |
| `--output` | `-o` | — | Output filename. **Required.** |
| `--plot` | — | — | Filename for UMAP plot (requires `--structure`). |
| `--structure` | `-S` | — | Column with chemical structure strings (for plotting). |
| `--input-representation` | `-R` | `smiles` | Structure format. |
| `--plot-sample` | `-n` | `20000` | Subsample size for UMAP. |
| `--plot-seed` | `-e` | `42` | Random seed for UMAP. |
| `--extras` | `-x` | — | Additional columns to colour in the UMAP plot. |
| `--start` | — | `0` | First row to process. |
| `--end` | — | end | Last row to process. |

#### Example

Tag the top 1 %, 5 %, and 10 % of cLogP and TPSA values, then save a UMAP plot coloured
by each percentile column:

```bash
$ eluent percentiles \
    hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train \
    --columns clogp tpsa \
    --percentiles 1 5 10 \
    --batch 128 \
    --cache ./cache \
    --output percentiles/tagged.csv \
    --plot percentiles/tagged-plot.png \
    --structure smiles
```

---

## Python API

### Splitting datasets

#### `split_dataset` — high-level function

```python
from datasets import load_dataset
from eluent.utils.splitting import split_dataset

ds = load_dataset(
    "scbirlab/fang-2023-biogen-adme",
    "scaffold-split",
    split="train",
)

# Scaffold split — 70 % train, 15 % validation, 15 % test (inferred)
split = split_dataset(
    ds,
    method="scaffold",
    structure_column="smiles",
    train=0.7,
    validation=0.15,
)
print(split)
# DatasetDict({
#     train:      Dataset(...),
#     validation: Dataset(...),
#     test:       Dataset(...),
# })
```

The `method` argument accepts `"random"`, `"scaffold"`, or `"faiss"`. Extra keyword
arguments are forwarded to the underlying grouping function.

```python
# Random split with reproducible seed
split = split_dataset(
    ds,
    method="random",
    structure_column="smiles",
    train=0.7,
    validation=0.15,
    seed=42,
)

# FAISS spectral split
split = split_dataset(
    ds,
    method="faiss",
    structure_column="smiles",
    n_neighbors=10,
    train=0.7,
    validation=0.15,
    seed=42,
    cache="./cache",
)
```

You can also pass a list of method dicts to chain grouping steps:

```python
split = split_dataset(
    ds,
    method=[{"method": "scaffold", "structure_column": "smiles", "train": 0.7, "validation": 0.15}],
)
```

#### `split_dataset` with k-fold cross-validation

When `kfolds > 1`, the function returns a **tuple** of `DatasetDict`s:

```python
folds = split_dataset(
    ds,
    method="scaffold",
    structure_column="smiles",
    train=0.7,
    validation=0.15,
    kfolds=5,
)
for i, fold in enumerate(folds):
    print(f"Fold {i}: {fold['train'].num_rows} train, {fold['validation'].num_rows} val")
```

#### `SplitDataset` — low-level class

`SplitDataset` wraps a dataset and provides separate `group()` and `split()` steps for
fine-grained control.

```python
from eluent.utils.splitting.splitter import SplitDataset

sd = SplitDataset(ds)

# Step 1: annotate each row with a group label
sd.dataset = sd.group(
    method="scaffold",
    structure_column="smiles",
    group_column="my_group",
)

# Step 2: pack groups into splits
result = sd.split(
    group_column="my_group",
    splits={"train": 0.7, "validation": 0.15, "test": 0.15},
    seed=42,
)
# result is a DatasetDict
```

Or use the combined helper:

```python
sd.group_and_split(
    method="scaffold",
    structure_column="smiles",
    train=0.7,
    validation=0.15,
)
result = sd.dataset
```

---

### Annotating percentiles

#### `percentiles` — high-level function

```python
from datasets import load_dataset
from eluent.utils.splitting.top_k import percentiles

ds = load_dataset(
    "scbirlab/fang-2023-biogen-adme",
    "scaffold-split",
    split="train",
    streaming=True,
)

tagged = percentiles(
    ds=ds,
    q={
        "clogp": [1, 5, 10],
        "tpsa":  [5],
    },
    compression=500,  # T-Digest centroids — higher is more accurate
    delta=1.0,        # buffer around percentile boundary
    cache="./cache",
)
# New boolean columns: clogp_top_1_pc, clogp_top_5_pc,
#                      clogp_top_10_pc, tpsa_top_5_pc
```

Use `reverse=True` to tag the _bottom_ percentiles instead:

```python
tagged = percentiles(
    ds=ds,
    q={"clogp": [5]},
    reverse=True,
)
# clogp_top_5_pc is True for the lowest 5 % of clogp values
```

#### `get_percentile` — single column, single threshold

For more direct control, call `get_percentile` after building a `TDigest` manually:

```python
from tdigest import TDigest
from eluent.utils.splitting.top_k import get_percentile

digest = TDigest(K=500)
for batch in ds.iter(batch_size=1024):
    digest.batch_update(batch["clogp"])

tagged = get_percentile(
    ds=ds,
    column="clogp",
    digest=digest,
    p=5.0,
    count=len(ds),
    delta=1.0,
    cache="./cache",
)
```
