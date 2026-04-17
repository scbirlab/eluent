# eluent

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/eluent/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eluent)
![PyPI](https://img.shields.io/pypi/v/eluent)

**eluent** is a Python library and command-line tool for chemistry-aware splitting of large datasets
for machine learning. It provides three grouping strategies — random, Murcko scaffold, and approximate
spectral clustering via FAISS — along with out-of-core percentile annotation. All operations work
on datasets that don't fit in memory, using [🤗 Hugging Face Datasets](https://huggingface.co/docs/datasets/)
for lazy processing and caching.

## Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Input data formats](#input-data-formats)
  - [Command-line interface](#command-line-interface)
    - [Dataset splitting](#dataset-splitting-eluent-split)
    - [Percentile annotation](#percentile-annotation-eluent-percentiles)
  - [Python API](#python-api)
    - [Splitting datasets](#splitting-datasets)
    - [Annotating percentiles](#annotating-percentiles)
- [Issues, problems, suggestions](#issues-problems-suggestions)
- [Documentation](#documentation)
- [Roadmap](#roadmap)

## Installation

### The easy way

You can install the pre-compiled version directly using `pip`.

```bash
$ pip install eluent
```

For GPU-accelerated FAISS splitting, install the optional GPU extra:

```bash
$ pip install eluent[splits_gpu]
```

### From source

Clone the repository, then `cd` into it. Then run:

```bash
$ pip install -e .
```

## Usage

### Input data formats

In all cases, the input dataset can be:

- A path to a **local file** in CSV, Parquet, Arrow, or HF Dataset directory format (format is
  inferred from the file extension).
- A **remote dataset** hosted on [🤗 Datasets Hub](https://huggingface.co/datasets), specified
  as a `hf://` URL:

```
hf://datasets/<owner>/<repo>~<config>:<split>
```

For example:

```
hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train
```

Output files are written in the format inferred from the extension of `--output`
(CSV, Parquet, Arrow, or HF Dataset directory).

### Command-line interface

**eluent** provides two sub-commands. Run `eluent --help` to see the top-level help:

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

#### Dataset splitting (`eluent split`)

Partition a dataset into train / validation / test splits using one of three grouping methods.
Groups are formed first, then packed into the requested split fractions using a bin-packing
algorithm. Pass `--seed` for randomised packing; omit it for fully deterministic packing.

```
usage: eluent split [-h] [--type {random,scaffold,faiss}] [--n-neighbors N_NEIGHBORS]
                    [--start START] [--end END] [--structure STRUCTURE]
                    [--input-representation {smiles,selfies,inchi,aa_seq}]
                    [--plot PLOT] [--plot-sample PLOT_SAMPLE] [--plot-seed PLOT_SEED]
                    [--extras [EXTRAS ...]] [--seed SEED] [--cache CACHE]
                    --output OUTPUT [--batch BATCH] [--kfolds KFOLDS]
                    [--train TRAIN] [--validation VALIDATION] [--test TEST]
                    input_file

positional arguments:
  input_file            Input file or hf:// URL.

options:
  --type {random,scaffold,faiss}
                        Splitting method. Default: scaffold
  --structure, -S       Column containing chemical structure strings.
  --input-representation, -R {smiles,selfies,inchi,aa_seq}
                        Structure string type. Default: smiles
  --train               Fraction of examples for training. Required.
  --validation          Fraction of examples for validation. Default: infer.
  --test                Fraction of examples for test. Default: infer.
  --kfolds, -K          Number of k-folds (overrides --validation). Default: 1
  --seed, -i            Random seed. Omit for deterministic bin-packing.
  --n-neighbors, -k     Nearest neighbours for FAISS grouping. Default: 10
  --batch, -b           Batch size. Default: 16
  --cache               Cache directory. Default: current directory.
  --output, -o          Output filename. Required.
  --plot                Filename for UMAP embedding plot.
  --plot-sample, -n     Subsample size for UMAP. Default: 20000
  --plot-seed, -e       Random seed for UMAP. Default: 42
  --extras, -x          Extra columns to colour in the UMAP plot.
  --start               First row to process. Default: 0
  --end                 Last row to process. Default: end of dataset.
  -h, --help            show this help message and exit
```

**Random splitting** — each molecule is assigned to a group by hashing its row index.
Groups are reproducible for the same `--seed`.

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

**Scaffold splitting** — molecules are grouped by their Murcko scaffold, so structurally similar
compounds always end up in the same split. This is the default and the recommended method for
chemistry ML benchmarks.

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

**FAISS spectral splitting** — Morgan fingerprints are computed for each molecule, a k-NN graph
is built using FAISS binary index (Hamming distance), and connected components of the graph become
groups. This produces splits where molecules in each component are all more similar to each other
than to molecules in other components.

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

**k-fold cross-validation** — use `--kfolds` to generate multiple train/validation folds.
The `--validation` fraction is split among folds; `--train` sets the remaining fraction.
Output files are written to `fold_0/`, `fold_1/`, … sub-directories of the output path.

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

#### Percentile annotation (`eluent percentiles`)

Add boolean columns to a dataset flagging which rows fall within the top-_k_ percentile of one
or more numeric columns. Uses a T-Digest quantile sketch, so the full dataset never needs to be
loaded into memory. A two-pass "definitely / maybe / definitely-not" strategy ensures exact counts.

```
usage: eluent percentiles [-h] [--columns [COLUMNS ...]] [--percentiles [PERCENTILES ...]]
                           [--reverse] [--compression COMPRESSION] [--delta DELTA]
                           [--start START] [--end END] [--cache CACHE] --output OUTPUT
                           [--batch BATCH] [--plot PLOT] [--structure STRUCTURE]
                           [--input-representation {smiles,selfies,inchi,aa_seq}]
                           [--plot-sample PLOT_SAMPLE] [--plot-seed PLOT_SEED]
                           [--extras [EXTRAS ...]]
                           input_file

positional arguments:
  input_file            Input file or hf:// URL.

options:
  --columns, -c         Columns to tag. Required.
  --percentiles, -p     Percentile thresholds. Default: 5
  --reverse, -r         Tag bottom percentiles instead of top.
  --compression, -z     T-Digest centroids (higher = more accurate). Default: 500
  --delta, -d           Buffer width around percentile cutoff. Default: 1.0
  --batch, -b           Batch size. Default: 16
  --cache               Cache directory. Default: current directory.
  --output, -o          Output filename. Required.
  --plot                Filename for UMAP embedding plot.
  --structure, -S       Structure column (required for --plot).
  --input-representation, -R {smiles,selfies,inchi,aa_seq}
                        Structure string type. Default: smiles
  --plot-sample, -n     Subsample size for UMAP. Default: 20000
  --plot-seed, -e       Random seed for UMAP. Default: 42
  --extras, -x          Extra columns to colour in the UMAP plot.
  --start               First row to process. Default: 0
  --end                 Last row to process. Default: end of dataset.
  -h, --help            show this help message and exit
```

Tag the top 1 %, 5 %, and 10 % of cLogP and TPSA values:

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

For each requested column and percentile, a new boolean column is added to the output dataset with
the name `<column>_top_<percentile>_pc` (e.g. `clogp_top_5_pc`).

### Python API

#### Splitting datasets

The high-level entry point is `split_dataset`, which accepts a 🤗 `Dataset` or `IterableDataset`
and returns a `DatasetDict`:

```python
from datasets import load_dataset
from eluent.utils.splitting import split_dataset

ds = load_dataset(
    "scbirlab/fang-2023-biogen-adme",
    "scaffold-split",
    split="train",
)

# Random split — 70 % train, 15 % validation, 15 % test
split = split_dataset(
    ds,
    method="random",
    structure_column="smiles",
    train=0.7,
    validation=0.15,
    seed=42,
)
# split["train"], split["validation"], split["test"]
```

Pass `method="scaffold"` or `method="faiss"` for chemistry-aware grouping:

```python
# Scaffold split
split = split_dataset(
    ds,
    method="scaffold",
    structure_column="smiles",
    train=0.7,
    validation=0.15,
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

For k-fold cross-validation, pass `kfolds > 1`. The function then returns a **tuple** of
`DatasetDict`s, one per fold:

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
    train_ds = fold["train"]
    val_ds   = fold["validation"]
```

For finer control, use the `SplitDataset` class directly:

```python
from eluent.utils.splitting.splitter import SplitDataset

sd = SplitDataset(ds)
sd.group_and_split(
    method="scaffold",
    structure_column="smiles",
    train=0.7,
    validation=0.15,
)
result = sd.dataset  # DatasetDict
```

#### Annotating percentiles

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
    q={"clogp": [1, 5, 10], "tpsa": [5]},
    compression=500,  # T-Digest centroids
    delta=1.0,        # buffer width around cutoff
    cache="./cache",
)
# New columns: clogp_top_1_pc, clogp_top_5_pc, clogp_top_10_pc, tpsa_top_5_pc
```

## Issues, problems, suggestions

Add to the [issue tracker](https://github.com/scbirlab/eluent/issues).

## Documentation

(To come at [ReadTheDocs](https://eluent.readthedocs.org).)

## Roadmap

The following features are planned for future releases:

- **Additional grouping methods** — pharmacophore-based, reaction-centre-based, and
  taxonomy-based grouping strategies, alongside the current random, scaffold, and FAISS methods.
- **Additional FAISS featurizers** — plug-in support for alternative fingerprint types
  (ECFP with configurable radius, MACCS keys, RDKit topological) and learned molecular
  embeddings, so the k-NN graph can be built on richer similarity measures.
- **Additional split strategies** — stratified splitting to preserve target-label class balance
  across splits; time-based splits for temporal datasets.
- **Multi-group / hierarchical splitting** — chain multiple grouping passes (e.g. scaffold
  then random) to produce nested or combined splits in a single command.
