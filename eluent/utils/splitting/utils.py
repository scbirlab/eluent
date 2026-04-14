"""Utilities used in splitting."""

from typing import Dict, Mapping, Union

from carabiner import print_err
from datasets import Dataset, IterableDataset
from tqdm.auto import tqdm


def annotate_split(
    x: Mapping[str, str], 
    key_to_split: Mapping[str, str],
    key_column: str,
    split_column: str = "split"
) -> Dict[str, ...]:
    """Tag each entry with its split label based on union-find component.

    Examples
    ========
    >>> import numpy as np
    >>> comp_map = {"A": "train", "B": "test"}
    >>> out = annotate_split({"col": ["B", "A"]}, key_to_split=comp_map, key_column="col")
    >>> out['split']
    ['test', 'train']

    """
    x[split_column] = [key_to_split.get(k) for k in x[key_column]]
    return x


def dataset_len(ds: Union[Dataset, IterableDataset]):
    if isinstance(ds, Dataset):
        return ds.num_rows
    elif isinstance(ds, IterableDataset):
        print_err("Looping through dataset to count rows: ", end="")
        for i, _ in enumerate(tqdm(ds)):
            pass
        return i
    else:
        raise ValueError(f"Input must be Dataset or IterableDataset but was `{type(ds)}`.")