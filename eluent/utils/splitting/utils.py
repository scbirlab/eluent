"""Utilities used in splitting."""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Mapping
from math import isfinite

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
else:
    Dataset, IterableDataset = Any, Any

from carabiner import print_err
from tqdm.auto import tqdm


def save_dataset(
    dataset,
    output: str
) -> None:
    output = str(output)
    print_err(
        "[INFO] Saving dataset:\n" 
        + str(dataset) 
        + "\n" 
        + f"at {output} as", 
        end=" ",
    )
    if output.removesuffix(".gz").endswith((".csv", ".tsv", ".txt")):
        print_err("CSV.")
        dataset.to_csv(
            output, 
            sep="," if output.removesuffix(".gz").endswith(".csv") else "\t",
            compression='gzip' if output.endswith(".gz") else None,
        )
    elif output.endswith(".json"):
        print_err("JSON.")
        dataset.to_json(output)
    elif output.endswith(".parquet"):
        print_err("Parquet.")
        dataset.to_parquet(output)
    elif output.endswith(".sql"):
        print_err("SQL.")
        dataset.to_sql(output)
    elif output.endswith(".hf"):
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output)
    else:
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output + ".hf")
        print_err(f"WARNING: Unsure what format to save as for filename {output}. Defaulted to Hugging Face dataset.")
    return None


def dataset_len(ds: Dataset | IterableDataset) -> int:
    from datasets import Dataset, IterableDataset
    if isinstance(ds, Dataset):
        return ds.num_rows
    elif isinstance(ds, IterableDataset):
        print_err("Looping through dataset to count rows: ", end="")
        for i, _ in enumerate(tqdm(ds)):
            pass
        return i + 1
    else:
        raise ValueError(f"Input must be Dataset or IterableDataset but was `{type(ds)}`.")


def validate_split_fractions(
    splits: Mapping[str, float | None],
    tolerance: float = 1e-6
) -> None:
    """Reject invalid requested split fractions.

    Fractions must be finite and between 0 and 1, and must not sum to more
    than 1 (within `tolerance`). `None` means "to be inferred" and is skipped.

    Examples
    ========
    >>> validate_split_fractions({"train": 0.7, "validation": None, "test": 0.3})
    >>> validate_split_fractions({"train": 0.9, "test": 0.3})
    Traceback (most recent call last):
        ...
    ValueError: Split fractions sum to 1.2, which is more than 1: {'train': 0.9, 'test': 0.3}
    >>> validate_split_fractions({"train": 0.7, "test": -0.1})
    Traceback (most recent call last):
        ...
    ValueError: Split fractions must be finite and between 0 and 1. Received: {'test': -0.1}
    >>> validate_split_fractions({"train": 1.2})
    Traceback (most recent call last):
        ...
    ValueError: Split fractions must be finite and between 0 and 1. Received: {'train': 1.2}

    """
    splits = {key: val for key, val in splits.items() if val is not None}
    invalid = {
        key: val for key, val in splits.items()
        if not (isfinite(val) and 0. <= val <= 1.)
    }
    if invalid:
        raise ValueError(
            f"Split fractions must be finite and between 0 and 1. Received: {invalid}"
        )
    total = sum(splits.values())
    if total > 1. + tolerance:
        raise ValueError(
            f"Split fractions sum to {total:.6g}, which is more than 1: {splits}"
        )
    return None


def check_new_columns(
    existing: Iterable[str],
    new: Iterable[str],
    what: str
) -> None:
    """Refuse to overwrite input columns with columns Eluent writes.

    Examples
    ========
    >>> check_new_columns(["id", "smiles"], ["train", "test"], "partition")
    >>> check_new_columns(["id", "test"], ["train", "test"], "partition")
    Traceback (most recent call last):
        ...
    ValueError: Input columns ['test'] clash with the partition columns Eluent writes, ['train', 'test']. Rename them in the input data.

    """
    clashes = sorted(set(existing) & set(new))
    if clashes:
        raise ValueError(
            f"Input columns {clashes} clash with the {what} columns Eluent "
            f"writes, {list(new)}. Rename them in the input data."
        )
    return None
