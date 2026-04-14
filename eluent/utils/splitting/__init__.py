

from typing import Union
from datasets import Dataset, IterableDataset
from .faiss import faiss_split
from .scaffold import scaffold_split

SPLIT_METHODS = {
    "scaffold": scaffold_split,
    "faiss": faiss_split,
}

def split_dataset(
    ds: Union[Dataset, IterableDataset],
    method: str,
    *args, **kwargs
):  
    try:
        return SPLIT_METHODS[method](ds, *args, **kwargs)
    except KeyError:
        raise KeyError(f"No method '{method}' for splitting")
