"""Utilities used in splitting."""

from typing import TYPE_CHECKING, Any, Dict, Mapping, Union

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
else:
    Dataset, IterableDataset = Any, Any

from carabiner import print_err
from tqdm.auto import tqdm


def dataset_len(ds: Union[Dataset, IterableDataset]):
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
