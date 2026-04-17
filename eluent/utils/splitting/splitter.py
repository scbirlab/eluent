"""Base splitting object."""
from typing import Callable, Iterable, Mapping, Optional, Tuple, Union
from functools import partial

from datasets import Dataset, DatasetDict, IterableDataset

from .decorators import process_splits
from .bin_packing import pack_bins
from .faiss import add_faiss_index
from .grouping import GROUPING_FUNCTIONS, annotate_split

DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_SEED: int = 42

class SplitDataset:

    def __init__(self, dataset, *args, **kwargs):
        if isinstance(dataset, (Dataset, IterableDataset)):
            self.dataset = DatasetDict({"train": dataset})
        elif isinstance(dataset, DatasetDict):
            self.dataset = dataset

    def group(
        self,
        method: Union[str, Callable],
        key: str = "train",
        group_column: str = "group",
        batch_size: int = DEFAULT_BATCH_SIZE,
        preprocess: Optional[Callable] = None,
        **kwargs
    ) -> 'SplitDataset':
        """Run group annotation over the dataset, returning an annotated copy."""
        if isinstance(method, str):
            method = method.casefold()
            try:
                group_fn = GROUPING_FUNCTIONS[method]
            except KeyError:
                raise ValueError(
                    f"Grouping method '{method}' is not implemented. Try one of these: {', '.join(GROUPING_FUNCTIONS)}"
                )
        elif isinstance(method, Callable):
            group_fn = method
        else:
            raise ValueError(
                f"Grouping method '{method}' must be a string or function. Instead, it was `{type(method)}`"
            )
        
        if method == "faiss" and preprocess is None:
            preprocess = partial(
                add_faiss_index,
                **kwargs
            )

        ds = self.dataset[key]
        if preprocess is not None and isinstance(preprocess, Callable):
            ds, info = preprocess(ds)
        else:
            ds, info = ds, None
        if info is not None and isinstance(info, dict):
            kwargs |= info
        if isinstance(ds, Dataset):
            desc = {"desc": f"Setting groups using {method}"}
        else:
            desc = {}
        return DatasetDict({
            key: ds.map(
                group_fn,
                fn_kwargs=kwargs | {"column": group_column},
                with_indices=True,
                batched=True,
                batch_size=batch_size,
                **desc,
            )
        } | {k: v for k, v in self.dataset.items() if k != key})

    def split(
        self,
        key: str = "train",
        group_column: str = "group",
        split_column: str = "split",
        deterministic: bool = True,
        seed: int = DEFAULT_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        splits: Optional[Mapping[str, float]] = None,
        **kwargs
    ) -> Union[DatasetDict, Tuple[DatasetDict, ...]]:

        """Split a dataset based on grouping.

        """
        ds = self.dataset[key]
        if group_column not in ds.column_names:
            raise KeyError(f"Grouping column '{group_column}' is not in data.")

        packed_bins = pack_bins(
            ds,
            group_column=group_column,
            splits=splits,
            batch_size=batch_size,
            deterministic=deterministic,
            seed=seed,
        )

        if isinstance(ds, Dataset):
            desc = {"desc": "Annotating splits"}
        else:
            desc = {}
        ds = ds.map(
            annotate_split,
            fn_kwargs={
                "key_to_split": packed_bins,
                "group_column": group_column,
                "column": split_column,
            },
            batched=True,
            batch_size=batch_size,
            **desc,
        )

        if isinstance(ds, Dataset):
            desc = {"desc": "Generating split datasets"}
        else:
            desc = {}
        if splits is None:
            splits = {}
        fold_keys = [k for k in splits if ":fold=" in k]
        nonfold_keys = set(k for k in splits if ":fold=" not in k)
        if len(fold_keys) > 0:
            return tuple(
                DatasetDict({
                    "train": ds.filter(lambda x, k=_fold: x[split_column] != k and x[split_column] in fold_keys, **desc),
                    "validation": ds.filter(lambda x, k=_fold: x[split_column] == k, **desc),
                }
                | {
                    _split: ds.filter(lambda x, k=_split: x[split_column] == k, **desc)
                    for _split in nonfold_keys
                }
                | {k: v for k, v in self.dataset.items() if k != key})
                for _fold in fold_keys
            )
        else:
            return DatasetDict({
                _split: ds.filter(lambda x, k=_split: x[split_column] == k, **desc)
                for _split in splits
            } | {k: v for k, v in self.dataset.items() if k != key})

    @process_splits
    def group_and_split(
        self,
        method: Union[str, Callable],
        key: str = "train",
        group_column: str = "group",
        split_column: str = "split",
        deterministic: bool = True,
        seed: int = DEFAULT_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        splits: Optional[Mapping[str, float]] = None,
        preprocess: Optional[Callable] = None,
        **kwargs
    ):
        self.dataset = self.group(
            key=key,
            group_column=group_column,
            method=method,
            batch_size=batch_size,
            preprocess=preprocess,
            **kwargs,
        )
        dataset = self.split(
            key=key,
            group_column=group_column,
            split_column=split_column,
            deterministic=deterministic,
            seed=seed,
            batch_size=batch_size,
            splits=splits,
            **kwargs,
        )
        self.dataset = dataset
        return self
