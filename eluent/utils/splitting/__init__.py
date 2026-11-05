"""Split datasets."""

from typing import TYPE_CHECKING, Any
from collections.abc import Mapping, Iterable

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset
else:
    Dataset, DatasetDict, IterableDataset = Any, Any, Any

from .splitter import SplitDataset

def split_dataset(
    ds: Dataset | IterableDataset,
    method: str | Mapping | Iterable,
    *args, **kwargs
) -> DatasetDict:  
    ds = SplitDataset(ds)
    if not isinstance(method, (list, tuple)):
        method = [method]
    if len(method) == 0:
        raise ValueError("No splitting methods provided!")
    for _method in method:
        if isinstance(_method, str):
            ds = ds.group_and_split(method=_method, **kwargs)
        elif isinstance(_method, Mapping):
            ds = ds.group_and_split(**_method)
    return ds.dataset
