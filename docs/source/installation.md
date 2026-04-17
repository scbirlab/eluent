# Installation

## The easy way

Install the pre-compiled version from PyPI:

```bash
$ pip install eluent
```

### GPU acceleration

For GPU-accelerated FAISS splitting (requires a CUDA-capable GPU):

```bash
$ pip install eluent[splits_gpu]
```

This replaces `faiss-cpu` with `faiss-gpu`.

## From source

Clone the repository, then `cd` into it and install in editable mode:

```bash
$ git clone https://github.com/scbirlab/eluent.git
$ cd eluent
$ pip install -e .
```

To also install development dependencies (pytest, flake8):

```bash
$ pip install -e ".[dev]"
```

## Requirements

**eluent** requires Python ≥ 3.11 and depends on:

| Package | Purpose |
|---------|---------|
| `datasets` ≥ 3.0 | Out-of-core dataset handling |
| `faiss-cpu` ≥ 1.11 | k-NN graph for spectral splitting |
| `schemist` ≥ 0.0.6 | Scaffold extraction and fingerprint computation |
| `tdigest` ≥ 0.5 | Out-of-core quantile sketches |
| `umap-learn` | UMAP visualisation of splits |
| `carabiner-tools` ≥ 0.0.5.post3 | CLI utilities |
| `numpy`, `scipy`, `tqdm` | Numerical utilities |
| `huggingface_hub` | Remote dataset access |
