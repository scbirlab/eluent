# eluent

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/scbirlab/eluent/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/eluent)
![PyPI](https://img.shields.io/pypi/v/eluent)

<img src="docs/source/_static/logo.svg" width="200" alt="Eluent logo">

**eluent** is a Python library and command-line tool for chemistry-aware splitting of large datasets for machine learning.

It supports random, Murcko scaffold, FAISS similarity-based, and group-aware partitioning, including disjoint splits across multiple grouping axes. Split definitions can be checkpointed and restored without recomputing group assignments or bin packing.

Eluent uses [Providence](https://github.com/scbirlab/providence) for dataset loading, provenance tracking, and source integrity verification, and [🤗 Hugging Face Datasets](https://huggingface.co/docs/datasets/) for scalable dataset processing.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Input data formats](#input-data-formats)
  - [Command-line interface](#command-line-interface)
    - [Dataset splitting](#dataset-splitting)
    - [Split checkpoints](#split-checkpoints)
    - [Exporting checkpoints](#exporting-checkpoints)
    - [Plotting](#plotting)
    - [Percentile annotation](#percentile-annotation)
  - [Python API](#python-api)
    - [Splitting datasets](#splitting-datasets)
    - [Saving and restoring checkpoints](#saving-and-restoring-checkpoints)
    - [Annotating percentiles](#annotating-percentiles)
- [Reproducibility](#reproducibility)
- [Issues, problems, suggestions](#issues-problems-suggestions)
- [Documentation](#documentation)
- [Roadmap](#roadmap)

## Installation

### From PyPI

Requires Python 3.11 or later.

```bash
python -m pip install eluent
```

For GPU-accelerated FAISS grouping, install the optional GPU dependencies on a supported platform:

```bash
python -m pip install 'eluent[splits_gpu]'
```

### From source

```bash
git clone https://github.com/scbirlab/eluent.git
cd eluent
git checkout feat/providence-split-checkpoints
python -m pip install -e .
```

Providence is installed automatically as a dependency.

Some releases of the transitive dependency `descriptastorus` may require installing `six` separately. If necessary:

```bash
python -m pip install six
```

## Quick Start

Create a small example dataset containing four chemical scaffolds, four assay batches, and eight measurements per scaffold–batch combination.

```bash
python - <<'PY'
import csv
from itertools import product

scaffolds = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCOCC1"]

with open("molecules.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["id", "smiles", "batch", "score"])
    for row, (smiles, batch, _) in enumerate(
        product(scaffolds, range(4), range(8))
    ):
        writer.writerow([row, smiles, batch, row])
PY
```

### 1. Split by chemical scaffold

Ensure molecules sharing a Murcko scaffold are assigned to the same partition.

```bash
eluent split molecules.csv \
    --type scaffold \
    --structure smiles \
    --train 0.75 --test 0.25 \
    --output demo/scaffold.csv
```

### 2. Create a reproducible split checkpoint

Save a split definition, including the source provenance and row assignments.

```bash
eluent split molecules.csv \
    --type scaffold \
    --structure smiles \
    --train 0.75 --test 0.25 \
    --checkpoint demo/scaffold.el
```

### 3. Restore and export the split

Recreate the partitioned datasets without rerunning the splitting algorithm.

```bash
eluent write demo/scaffold.el \
    --output demo/restored.csv
```

### 4. Evaluate generalisation across multiple axes

Hold out entire chemical scaffolds and assay batches simultaneously.

```bash
eluent split molecules.csv \
    --group-columns smiles batch \
    --train 1 --kfolds 2 \
    --checkpoint demo/multiway.el
```

Two folds on each axis produce four cross-validation evaluations.

### 5. Annotate top-performing compounds

Flag observations in the highest-scoring 10%.

```bash
eluent percentiles molecules.csv \
    --columns score \
    --percentiles 10 \
    --delta 0.25 \
    --output demo/tagged.csv
```

## Usage

### Input data formats

Eluent delegates data loading to [Providence](https://github.com/scbirlab/providence).

Supported inputs include local CSV, TSV, TXT, JSON and Parquet files, Hugging Face datasets, and Providence-supported checkpoint references. The Python API also accepts in-memory datasets and DataFrames.

For Hugging Face Hub datasets, use:

```text
hf://datasets/<owner>/<repo>@<revision>~<config>:<split>
```

For example:

```text
hf://datasets/scbirlab/fang-2023-biogen-adme~scaffold-split:train
```

Revision, configuration, and split selectors are optional where supported.

For a subset of rows, use `--start` and `--end`, with zero-based indexing and an exclusive end:

```bash
eluent split molecules.csv \
    --start 100 --end 1000 \
    --type random \
    --train 0.8 --test 0.2 \
    --checkpoint demo/subset.el
```

### Command-line interface

Eluent provides four commands:

| Command | Purpose |
|---|---|
| `eluent split` | Generate partitions and optionally checkpoint them |
| `eluent write` | Export tables from a saved split checkpoint |
| `eluent plot` | Visualise chemical splits or percentile annotations |
| `eluent percentiles` | Annotate numeric columns by percentile |

Run:

```bash
eluent --help
eluent split --help
eluent write --help
eluent plot --help
eluent percentiles --help
```

Commands accept `--cache` where applicable. Cache configuration is delegated to Providence.

### Dataset splitting

Eluent assigns groups to partitions using bin packing. Requested split fractions are targets; achieved fractions depend on the number and size of groups.

Specify at least `--train`. For ordinary partitioning, omitting `--validation` and `--test` infers the remaining fractions. Explicit zero fractions are respected.

Omitting `--seed` uses deterministic bin packing. Supplying `--seed` enables reproducible randomized packing; different seeds may produce different assignments.

The same seed controls random grouping where applicable.

#### Random splitting

Assign observations to random groups using row-derived hashes.

```bash
eluent split molecules.csv \
    --type random \
    --train 0.7 --validation 0.15 --test 0.15 \
    --seed 42 \
    --output demo/random.csv
```

Keep input row order fixed for reproducible random group assignments.

#### Scaffold splitting

Group molecules by Bemis–Murcko scaffold.

```bash
eluent split molecules.csv \
    --type scaffold \
    --structure smiles \
    --train 0.75 --test 0.25 \
    --output demo/scaffold.csv
```

Molecules with identical scaffolds remain in the same partition. Ringless molecules are grouped by canonical SMILES rather than all sharing an empty scaffold.

For supported non-SMILES chemical representations, supply `--input-representation`.

#### FAISS splitting

Compute molecular fingerprints, identify neighbours using FAISS, and group molecules into connected components of the nearest-neighbour graph.

```bash
eluent split molecules.csv \
    --type faiss \
    --structure smiles \
    --n-neighbors 2 \
    --train 0.75 --test 0.25 \
    --cache ./cache \
    --output demo/faiss.csv
```

Connected components are assigned as indivisible groups. Large components can make requested split fractions difficult to achieve.

The default number of neighbours is 10. FAISS processing may require substantial working memory.

#### Splitting by existing groups

Use `--group-columns` to specify columns already containing group labels. This overrides `--type`.

```bash
eluent split molecules.csv \
    --group-columns batch \
    --train 0.75 --test 0.25 \
    --output demo/batch.csv
```

All observations sharing a batch remain in the same partition.

#### Multiway splitting

Multiple grouping axes can be supplied together:

```bash
eluent split molecules.csv \
    --group-columns smiles batch \
    --train 0.75 --test 0.25 \
    --seed 42 \
    --output demo/multiway.csv
```

Each axis is partitioned independently. An observation enters a fixed partition only when the assignments agree across all axes.

For example, the training set contains observations whose scaffold and batch are both assigned to training.

Rows with conflicting assignments are excluded from the corresponding fixed partitions. Consequently, achieved fractions may be substantially smaller than the requested fractions, which apply independently to each axis.

This enables evaluation of generalisation along multiple dimensions, such as chemical scaffold, organism, or protein target.

#### Cross-validation

Set `--kfolds` to generate cross-validation partitions.

```bash
eluent split molecules.csv \
    --group-columns batch \
    --train 0.75 \
    --kfolds 3 \
    --seed 42 \
    --checkpoint demo/cv.el
```

Here, 75% of groups are assigned to the cross-validation pool and the remainder to a fixed test partition.

Each evaluation holds out one fold for validation and uses the remaining folds for training.

For multiway cross-validation, `K` folds across `D` grouping axes produce `K**D` evaluations.

```bash
eluent split molecules.csv \
    --group-columns smiles batch \
    --train 1 \
    --kfolds 2 \
    --checkpoint demo/multiway-cv.el
```

Each evaluation contains:

- `train`: observations held out on no axes.
- `validation`: observations held out on every axis.
- `validation_partial`: observations held out on some, but not all, axes.

When requested, a fixed `test` partition is reused across evaluations.

The full training and validation sets are group-disjoint on every axis. Partial validation observations can be used to investigate intermediate generalisation regimes.

### Split checkpoints

A split checkpoint stores a reproducible partition definition independently of its exported tables.

```bash
eluent split molecules.csv \
    --type scaffold \
    --structure smiles \
    --train 0.75 --test 0.25 \
    --checkpoint demo/scaffold.el
```

To save both the checkpoint and partition tables:

```bash
eluent split molecules.csv \
    --type scaffold \
    --structure smiles \
    --train 0.75 --test 0.25 \
    --checkpoint demo/scaffold.el \
    --output demo/scaffold.csv
```

A checkpoint contains:

```text
scaffold.el/
├── schema.json
├── info.json
├── partition.parquet
└── resolved.pv/
    └── ...
```

The components have distinct responsibilities:

| Component | Contents |
|---|---|
| `schema.json` | Checkpoint format version and component paths |
| `info.json` | Partition names, counts, fractions and grouping metadata |
| `partition.parquet` | Row-level partition assignments |
| `resolved.pv/` | Nested Providence dataset checkpoint |

Providence manages source provenance and integrity verification. Eluent records the resulting split assignments.

For existing local or remote sources, the checkpoint may retain references rather than copying the entire dataset. To request embedding source data, use:

```bash
eluent split molecules.csv \
    --type random \
    --train 0.8 --test 0.2 \
    --checkpoint demo/portable.el \
    --force_write_data
```

Checkpoint restoration does not rerun the grouping or bin-packing algorithm.

### Exporting checkpoints

Use `eluent write` to materialise saved partitions into tables.

```bash
eluent write demo/scaffold.el \
    --output demo/restored.csv
```

For Parquet output:

```bash
eluent write demo/scaffold.el \
    --output demo/restored.parquet
```

The output path supplies the base directory and file format. For example, `--output demo/restored.parquet` produces files under:

```text
demo/restored/
└── fold_x/
    ├── train.parquet
    └── test.parquet
```

Cross-validation checkpoints generate one directory per evaluation. Generated fold names are retained so that the saved partition definitions can be reconstructed.

### Plotting

Plotting is a separate command, allowing saved checkpoints and annotated tables to be visualised without repeating the original computation.

#### Plot split assignments

```bash
eluent plot demo/scaffold.el \
    --mode split \
    --structure smiles \
    --output demo/scaffold.png
```

#### Plot percentile annotations

```bash
eluent plot demo/tagged.csv \
    --mode percentiles \
    --structure smiles \
    --output demo/percentiles.png
```

Eluent generates a sampled UMAP embedding of chemical structures, coloured by split assignment or percentile annotation.

Use `--plot-sample` (default 20,000), `--seed`, and `--extras` to control sampling, embedding reproducibility, and additional colouring variables.

Plotting requires a structure column and may involve substantial computation for large datasets.

### Percentile annotation

Use `eluent percentiles` to flag observations in the top or bottom percentages of numeric columns.

```bash
eluent percentiles molecules.csv \
    --columns score \
    --percentiles 1 5 10 \
    --compression 500 \
    --delta 0.25 \
    --output demo/percentiles.csv
```

Eluent uses a T-Digest sketch to estimate percentile thresholds, then refines selection near each cutoff.

The default percentile is 5%. `--compression` controls sketch resolution and `--delta` determines the refinement buffer in the column's units.

Output columns follow:

```text
<column>_top_<percentile>_pc
```

For example:

```text
score_top_1_pc
score_top_5_pc
score_top_10_pc
```

Use `--reverse` to select the lowest values.

The refinement procedure targets the requested number of observations, but results near a cutoff may differ from a complete exact sort. Output rows may be reordered.

## Python API

### Splitting datasets

The principal entry points are `split_dataset()` and `SplitDataset`.

Both accept Providence-compatible data inputs.

```python
from datasets import load_dataset
from eluent.utils.splitting import split_dataset

data = load_dataset(
    "csv",
    data_files="molecules.csv",
    split="train",
)

result = split_dataset(
    data,
    method="scaffold",
    structure_column="smiles",
    train=0.75,
    test=0.25,
)

print(result.split_info.counts)
```

`split_dataset()` returns a `SplitDataset` instance, containing three related objects:

| Attribute | Description |
|---|---|
| `.dataset` | Annotated source dataset, represented as a DatasetDict |
| `.split_dataset` | Mapping of evaluations to partitioned DatasetDict objects |
| `.split_info` | Grouping, partition and count metadata |

Access the ordinary partitions using:

```python
train = result.split_dataset["fold_x"]["train"]
test = result.split_dataset["fold_x"]["test"]
```

#### Other splitting methods

```python
random_split = split_dataset(
    data,
    method="random",
    train=0.7,
    validation=0.15,
    test=0.15,
    seed=42,
)

faiss_split = split_dataset(
    data,
    method="faiss",
    structure_column="smiles",
    n_neighbors=2,
    train=0.75,
    test=0.25,
    cache="./cache",
)
```

#### Multiway splitting

Supply existing grouping columns using `group_column`.

```python
result = split_dataset(
    data,
    method=None,
    group_column=("smiles", "batch"),
    train=1.0,
    kfolds=2,
)

for fold_name, fold in result.split_dataset.items():
    print(
        fold_name,
        len(fold["train"]),
        len(fold["validation"]),
        len(fold["validation_partial"]),
    )
```

#### Using `SplitDataset` directly

For more control, instantiate the class explicitly:

```python
from eluent.utils.splitting import SplitDataset

splitter = SplitDataset(data)

splitter.group_and_split(
    method="scaffold",
    structure_column="smiles",
    train=0.75,
    test=0.25,
)

print(splitter.split_info.fractions)
```

`group_and_split()` annotates and partitions the dataset, updates the wrapper, and returns `self`.

The lower-level `split()` method accepts existing grouping columns and explicit split weights. It returns the annotated dataset, partition mapping, and split metadata as a tuple.

### Saving and restoring checkpoints

```python
from eluent.utils.splitting import SplitDataset

splitter = SplitDataset("molecules.csv")

splitter.group_and_split(
    method="scaffold",
    structure_column="smiles",
    train=0.75,
    test=0.25,
)

splitter.save("demo/scaffold.el")
```

Restore the exact split assignments:

```python
restored = SplitDataset.load(
    "demo/scaffold.el",
    cache_dir="./cache",
)

train = restored.split_dataset["fold_x"]["train"]
test = restored.split_dataset["fold_x"]["test"]
```

Export the restored partitions:

```python
restored.write_tables(
    "demo/restored",
    format="parquet",
)
```

For a self-contained checkpoint where source data must be embedded:

```python
splitter.save(
    "demo/portable.el",
    save_source_data=True,
)
```

### Annotating percentiles

The Python API supports Hugging Face `Dataset` and `IterableDataset` inputs.

```python
from eluent.utils.splitting.top_k import percentiles

tagged = percentiles(
    ds=data,
    q={"score": [1, 5, 10]},
    compression=500,
    delta=0.25,
    cache_dir="./cache",
)
```

The result contains boolean annotation columns such as `score_top_5_pc`.

## Reproducibility

Eluent and Providence divide reproducibility responsibilities.

**Providence** manages dataset resolution, source metadata, checkpointing, and source integrity. Local source files and embedded checkpoint data can be verified using checksums. Hugging Face revisions are pinned to resolved commits.

**Eluent** manages group assignments, partition generation, cross-validation definitions, and saved row-level membership.

Together, they allow an experiment to retain both the identity of its input data and the precise partition definitions used for training and evaluation.

For reproducible workflows:

1. Keep dataset row order stable, especially for random grouping.
2. Record the grouping method, parameters, and seed.
3. Save the split checkpoint alongside model configuration and results.
4. Restore saved assignments rather than recomputing partitions when repeating an experiment.

A checkpoint referencing an external source requires that source to remain accessible.

## Issues, problems, suggestions

Please use the [GitHub issue tracker](https://github.com/scbirlab/eluent/issues).

## Documentation

Additional documentation is planned at [ReadTheDocs](https://eluent.readthedocs.org).

## Roadmap

Potential future improvements include:

- **Additional grouping methods** for pharmacophores, reaction centres, and taxonomy.
- **Additional FAISS featurizers** including configurable ECFP, MACCS keys, RDKit topological fingerprints, and learned embeddings.
- **Additional split strategies** to preserve target-label distributions or respect time order.
- **Hierarchical splitting** to chain multiple grouping and partitioning operations.
- **Extended checkpoint validation** for additional dataset and partition transformations.
