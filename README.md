# eluent

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/scbirlab/eluent/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eluent)
![PyPI](https://img.shields.io/pypi/v/eluent)

<img src="docs/source/_static/logo.svg" width="200">

**eluent** is a Python library and command-line tool for chemistry-aware splitting of large datasets
for machine learning. It can group rows randomly, by Murcko scaffold, by connected components of
a FAISS nearest-neighbor graph, or by existing group labels. It also supports holdouts across
multiple grouping axes and adds percentile flags to numeric columns. It uses
[🤗 Hugging Face Datasets](https://huggingface.co/docs/datasets/) for disk-backed processing and
caching. Some steps, including FAISS indexing, also need in-memory working data.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
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

Install the published package with Python 3.11 or later.

```bash
python -m pip install eluent
```

For GPU-accelerated FAISS grouping through the Python API, install the optional GPU extra
on a platform with a compatible `faiss-gpu` build and pass `gpu=True` to the FAISS method.

```bash
python -m pip install 'eluent[splits_gpu]'
```

### From source

Install the branch to use the multiway examples below.

```bash
git clone --branch feat/multiway-split https://github.com/scbirlab/eluent.git
cd eluent
python -m pip install -e . six
```

`six` is included because some releases of the transitive dependency
`descriptastorus` import it without declaring it. If pip attempts to build
`llvmlite` from source, retry with `--only-binary=numba,llvmlite` to select
compatible wheels where available.

## Quick Start

After installing from source, generate a local CSV with four scaffolds, four assay batches,
and eight measurements per scaffold–batch pair.

```bash
python - <<'PY'
import csv
from itertools import product

scaffolds = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCOCC1"]
with open("molecules.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["id", "smiles", "batch", "score"])
    for row, (smiles, batch, _) in enumerate(product(scaffolds, range(4), range(8))):
        writer.writerow([row, smiles, batch, row])
PY
```

Split by Murcko scaffold to hold out entire scaffolds.

```bash
eluent split molecules.csv --structure smiles --train 0.75 --test 0.25 \
  --output demo/scaffold.csv
```

Hold out both scaffold and assay batch. Two folds per axis produce four evaluations.

```bash
eluent split molecules.csv --group-columns smiles batch --train 1 --kfolds 2 \
  --output demo/multiway.csv
```

Flag the highest-scoring 10% of rows.

```bash
eluent percentiles molecules.csv --columns score --percentiles 10 --delta 0.25 \
  --output demo/tagged.csv
```

## Usage

### Input data formats

Pass a local CSV, TSV or tab-delimited TXT file (optionally gzip-compressed), JSON, Parquet,
Arrow, or XML file, or a Hugging Face dataset directory with a `.hf` suffix. The loader infers
the format from the suffix. You can also pass a remote dataset from the
[🤗 Datasets Hub](https://huggingface.co/datasets).

```text
hf://datasets/<owner>/<repo>~<config>:<split>
```

For example:

```text
hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train
```

The config and split are optional. You can add `@<revision>` after the repository name.
If the Hub returns several splits because you omitted the split, eluent concatenates them.
Use `--start` and `--end` to select a zero-based row range with an exclusive end.

For file output, use CSV, TSV/TXT (also with gzip), JSON, Parquet, or a
Hugging Face dataset directory. Use `.hf` for an explicit dataset directory. For an unrecognized
output suffix, eluent appends `.hf` and saves a dataset directory.

### Command-line interface

**eluent** provides two subcommands. Run these for the full option lists.

```bash
eluent --help
eluent --version
eluent split --help
eluent percentiles --help
```

Both commands accept `--cache`. It defaults to `ELUENT_CACHE` if set, then to the platform's
user cache directory for eluent. Both can also plot a sampled UMAP embedding with `--plot`
and `--structure`. Use `--plot-sample` (default 20,000), `--plot-seed` (default 42), and
`--extras` to control the sample, seed, and additional columns used for coloring.

#### Dataset splitting (`eluent split`)

eluent assigns whole groups to splits using bin packing. Fractions are targets, so the achieved
sizes depend on group sizes. Provide `--train`. For a single split, passing `--train` alone assigns
the remainder to validation. Supplying either `--validation` or `--test` lets eluent infer the
other. Use fractions that sum to 1 when specifying all three.

Pass `--seed` for deterministic greedy packing. Omitting it uses randomized packing with an
internal seed of 42. The same seed also controls the row hashes for random grouping. Use `--batch`
(default 16) to change the group annotation and packing batch size.

The output path supplies the directory, filename stem, and format. For example,
`--output demo/split.csv` writes `demo/fold_0/split_train.csv` and one file for each other split.
Additional evaluations use `fold_1/`, `fold_2/`, and so on. A plot requested as
`demo/split.png` is saved with a fold suffix such as `demo/split-fold_0.png`.

**Random splitting** hashes row indices into pseudo-unique groups. It requires no structure
column. Keep the input row order fixed for repeatable group assignments.

```bash
eluent split molecules.csv --type random --train 0.7 --validation 0.15 \
  --seed 42 --output demo/random.csv
```

**Scaffold splitting** is the default. Rows with the same Murcko scaffold stay together.
Supply the structure column with `--structure`. For non-SMILES input, choose a representation
supported by the requested conversion with `--input-representation`.

```bash
eluent split molecules.csv --type scaffold --structure smiles \
  --train 0.75 --test 0.25 --output demo/scaffold.csv
```

**FAISS splitting** computes Morgan fingerprints from SMILES and groups rows by connected
components of a binary nearest-neighbor graph using Hamming distance. A chain of neighbor links
can join molecules into a large component, which can make target fractions hard to reach.
`--n-neighbors` defaults to 10. Component labels are cached for repeat runs.

```bash
eluent split molecules.csv --type faiss --structure smiles --n-neighbors 2 \
  --train 0.75 --test 0.25 --cache ./cache --output demo/faiss.csv
```

**Existing groups** are selected with `--group-columns`, which takes precedence over
`--type`. Supply one column to keep each group intact. With two or more distinct columns,
eluent packs each axis independently and keeps a row in a split when every axis assigns
it to that split. Rows with conflicting assignments are omitted from fixed splits.
Fractions apply per axis before intersection, so the retained splits can be smaller than
the requested fractions of the input.

```bash
eluent split molecules.csv --group-columns smiles batch --train 0.75 --test 0.25 \
  --seed 42 --output demo/holdout.csv
```

**k-fold cross-validation** uses `--train` as the size of the entire cross-validation pool.
With `--kfolds K` greater than 1, eluent ignores `--validation` and defaults the fixed test
fraction to the remainder of the input. Each single-axis evaluation holds out one fold for
validation, trains on the other folds, and reuses the fixed test split when present.

```bash
eluent split molecules.csv --group-columns batch --train 0.75 --kfolds 3 \
  --seed 42 --output demo/cv.csv
```

For multiway cross-validation, `K` folds on each of `D` axes produce `K**D` evaluations
in Cartesian-product order of the supplied axes. Within the cross-validation pool, each
evaluation has three parts.

- `train` contains rows held out on zero axes.
- `validation` contains rows held out on every axis.
- `validation_partial` contains rows held out on some axes.

The training and full validation sets share no group labels on any axis. Each row in the
cross-validation pool appears in full validation exactly once across the evaluations. When requested,
the fixed `test` split is the intersection of test groups on all axes and is reused in every
evaluation. A row assigned to the fixed test partition on only some axes is outside the
cross-validation pool and the fixed test split.

#### Percentile annotation (`eluent percentiles`)

Add boolean columns flagging the top percentages of one or more numeric columns.
eluent builds a T-Digest sketch in batches, then refines the selection near each cutoff.
It targets the ceiling of the requested percentage of rows. When several distinct scores fall in the cutoff buffer,
the refinement can select a different set of rows than an exact sort would. Output rows may
be reordered.

```bash
eluent percentiles molecules.csv --columns score --percentiles 1 5 10 \
  --compression 500 --delta 0.25 --output demo/percentiles.csv
```

Use `--columns` to supply at least one numeric column and `--percentiles` to override
the default of 5. `--compression` controls the sketch and `--delta` sets the cutoff buffer
in the column's units (CLI default 1.0). The output names are `<column>_top_<percentile>_pc`, for example
`score_top_5_pc`. The suffix uses the integer part of the threshold, so use distinct integer
thresholds to keep names unique. `--reverse` targets the lowest values and uses the same
column naming convention.

### Python API

#### Splitting datasets

Use `split_dataset` with a Hugging Face `Dataset`. It returns a `DatasetDict` for a
single evaluation and a tuple of `DatasetDict` objects for cross-validation.

```python
from datasets import load_dataset
from eluent.utils.splitting import split_dataset

ds = load_dataset("csv", data_files="molecules.csv", split="train")

random_split = split_dataset(
    ds, method="random", train=0.7, validation=0.15, seed=42
)
scaffold_split = split_dataset(
    ds, method="scaffold", structure_column="smiles", train=0.75, test=0.25
)
faiss_split = split_dataset(
    ds, method="faiss", structure_column="smiles", n_neighbors=2,
    train=0.75, test=0.25, cache="./cache",
)
```

For single-axis cross-validation, pass `kfolds` greater than 1 to any of these methods.

Pass `method=None` to use existing group labels. Use a string for single-axis splitting
and a list or tuple of distinct column names for multiway splitting.

```python
folds = split_dataset(
    ds,
    method=None,
    group_column=("smiles", "batch"),
    train=1.0,
    kfolds=2,
)
for fold in folds:
    print(len(fold["train"]), len(fold["validation"]), len(fold["validation_partial"]))
```

For finer control, use the `SplitDataset` class directly. `group_and_split` updates the
wrapper and returns it. Its optional `splits` mapping adds or overrides inferred split
weights. `split` partitions existing groups using an explicit mapping of
split names to fractions and returns the partitions directly. When initialized with a
`DatasetDict`, the wrapper splits the selected `key` (default `"train"`) and passes through
the other entries. Use distinct names for the generated splits and the entries you want to keep.

```python
from eluent.utils.splitting.splitter import SplitDataset

sd = SplitDataset(ds)
sd.group_and_split(
    method="scaffold", structure_column="smiles", train=0.75, test=0.25
)
result = sd.dataset
```

#### Annotating percentiles

The percentile API accepts a `Dataset` or an `IterableDataset`. The latter is returned as
an iterable after refinement, which materializes data for sorting and concatenation.

```python
from eluent.utils.splitting.top_k import percentiles

tagged = percentiles(
    ds=ds,
    q={"score": [1, 5, 10]},
    compression=500,
    delta=0.25,
    cache="./cache",
)
# New columns: score_top_1_pc, score_top_5_pc, score_top_10_pc
```

## Issues, problems, suggestions

Add to the [issue tracker](https://github.com/scbirlab/eluent/issues).

## Documentation

(To come at [ReadTheDocs](https://eluent.readthedocs.org).)

## Roadmap

The following features are planned for future releases:

- **Additional grouping methods** for pharmacophores, reaction centers, and taxonomy.
- **Additional FAISS featurizers** such as configurable ECFP, MACCS keys, RDKit topological
  fingerprints, and learned molecular embeddings.
- **Additional split strategies** for preserving target-label balance or respecting time order.
- **Hierarchical splitting** to chain grouping passes, such as scaffold followed by random
  splitting, into nested splits.
