"""Base splitting object."""
from typing import Callable, Iterable, Mapping, Optional, Union
from collections.abc import Callable, Iterable, Mapping
from collections import defaultdict
from functools import partial
from itertools import product
from math import isfinite

from datasets import Dataset, DatasetDict, IterableDataset

from .decorators import process_splits
from .bin_packing import pack_bins
from .faiss import add_faiss_index
from .grouping import GROUPING_FUNCTIONS, annotate_split, annotate_split_invert, annotate_split_invert_fixed

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
        if method == "faiss":
            group_kwargs = {
                "component_labels": info["component_labels"],
            }
        else:
            group_kwargs = kwargs
            if info is not None and isinstance(info, dict):
                group_kwargs |= info
        if isinstance(ds, Dataset):
            desc = {"desc": f"Setting groups using {method}"}
        else:
            desc = {}
        return DatasetDict({
            key: ds.map(
                group_fn,
                fn_kwargs=group_kwargs | {"column": group_column},
                with_indices=True,
                batched=True,
                batch_size=batch_size,
                **desc,
            )
        } | {k: v for k, v in self.dataset.items() if k != key})

    @staticmethod
    def _build_holdout_dict(
        split_keys: Iterable[str],
        group_to_split: Mapping,
        prefix: str = "fold=",
        group_column: Iterable[str] = ("group", ),
        fixed: bool = False
    ):
        new_fold_keys = []
        split_to_groups = {}
        for i, ijk in enumerate(product(
            range(len(split_keys)), 
            repeat=len(group_column),
        )):
            ijk_str = "-".join(map(str, ijk))
            if not fixed:
                fold_name = f"{prefix}{ijk_str}_{i}"
            else:
                _i = ijk[0]
                if ijk.count(_i) != len(ijk):
                    continue
                fold_name = split_keys[_i]
            holdout_groups = {}
            for holdout_key_idx, (column, packed_bins) in zip(ijk, group_to_split.items()):
                holdout_key = split_keys[holdout_key_idx]
                holdout_groups[column] = [
                    g for g, s in packed_bins.items() 
                    if s == holdout_key
                ]
            split_to_groups[fold_name] = tuple(
                set(holdout_groups[col])
                for col in group_column
            )
            new_fold_keys.append(fold_name)
        return split_to_groups, tuple(new_fold_keys)

    def _split_multiway(
        self,
        key: str = "train",
        group_column: Iterable[str] = "group",
        deterministic: bool = True,
        seed: int = DEFAULT_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        splits: Mapping[str, float] | None = None,
        **kwargs
    ):
        """Partition each axis independently, retaining only split intersections.

        Mixed assignments are excluded. Fractions apply to each axis before
        intersection, not to the sizes of the returned datasets. 
        With k folds and d axes, return k**d evaluations in Cartesian-product order.
        Existing DatasetDict entries are passed through without re-splitting.

        """
        ds = self.dataset[key]
        if not group_column or len(set(group_column)) != len(group_column):
            raise ValueError("Provide distinct grouping columns.")

        for column in group_column:
            if column not in ds.column_names:
                raise KeyError(f"Grouping column '{column}' is not in data: {ds.column_names}.")
            
            if column == split_column:
                raise ValueError(f"Output split column name '{split_column}' must differ from grouping columns.")
        
        if not splits or any([
            any(not isfinite(v) or v < 0. for v in splits.values()),
            sum(splits.values()) <= 0,
        ]):
            raise ValueError(
                "Provide nonnegative split fractions with a positive total. "
                f"Received: {splits}"
            )
        
        total = sum(splits.values())
        fractions = {k: v / total for k, v in splits.items()}
        group_to_split = {}
        for column in group_column:
            group_to_split[column] = pack_bins(
                ds, 
                group_column=column, 
                splits=fractions,
                batch_size=batch_size, 
                deterministic=deterministic, 
                seed=seed,
            )
        fold_keys = [k for k in fractions if ":fold=" in k]
        nonfold_keys = [k for k in fractions if ":fold=" not in k]
        if fold_keys and set(nonfold_keys) & {"train", "validation", "validation_partial"}:
            raise ValueError("Fixed splits cannot be named train/validation in cross-validation.")
        
        split_to_groups_fixed, new_fixed_keys = self._build_holdout_dict(
            split_keys=nonfold_keys,
            group_to_split=group_to_split,
            group_column=group_column,
            fixed=True,
        )
        if isinstance(ds, Dataset):
            desc = {"desc": "Annotating splits"}
        else:
            desc = {}
        ds = ds.map(
            annotate_split_invert_fixed,
            fn_kwargs={
                "split_to_group": split_to_groups_fixed,
                "group_column": group_column,
            },
            batched=True,
            batch_size=batch_size,
            **desc,
        )
        fixed_ds = {
            _split: ds.filter(lambda x, k=_split: x[k], **desc)
            for _split in nonfold_keys
        }
        if len(fold_keys) > 0:
            split_to_groups, new_fold_keys = self._build_holdout_dict(
                split_keys=fold_keys,
                group_to_split=group_to_split,
                group_column=group_column,
                prefix=f"{key}:fold=",
            )
            fold_key_set = set(fold_keys)
            cv_groups = tuple(
                {
                    group
                    for group, split_name in group_to_split[column].items()
                    if split_name in fold_key_set
                }
                for column in group_column
            )

            ds = ds.map(
                annotate_split_invert,
                fn_kwargs={
                    "split_to_group": split_to_groups,
                    "group_column": group_column,
                    "cv_groups": cv_groups,
                },
                batched=True,
                batch_size=batch_size,
                **desc,
            )
            return tuple(
                DatasetDict({
                    _split: ds.filter(
                        lambda x, k=f"{_fold}|{_split}": x.get(k, False),
                        **desc,
                    )
                    for _split in (
                        "train", 
                        "validation", 
                        "validation_partial",
                    )
                }
                | fixed_ds
                | {k: self.dataset[k] for k in self.dataset if k != key})
                for _fold in new_fold_keys
            )
        else:
            return DatasetDict(fixed_ds | {
                k: self.dataset[k] for k in self.dataset if k != key
            })


    def split(
        self,
        key: str = "train",
        group_column: str | Iterable[str] = "group",
        split_column: str = "split",
        deterministic: bool = True,
        seed: int = DEFAULT_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        splits: Mapping[str, float] | None = None,
        **kwargs
    ) -> Union[DatasetDict, tuple[DatasetDict, ...]]:

        """Split a dataset based on grouping.

        ``group_column`` may be a column name or a sequence of independent
        grouping axes. Multiple axes use intersections: rows with conflicting
        split assignments are excluded. 
        Fold combinations are enumerated in Cartesian-product order; 
        each returned evaluation is group-disjoint
        on every axis. 
        Fractions apply per axis, before intersection.

        """
        ds = self.dataset[key]
        if not isinstance(group_column, str):
            if isinstance(group_column, (list, tuple)):
                return self._split_multiway(
                    key=key,
                    group_column=tuple(group_column),
                    split_column=split_column,
                    deterministic=deterministic,
                    seed=seed,
                    batch_size=batch_size,
                    splits=splits,
                    **kwargs,
                )
            else:
                raise ValueError(f"Grouping column '{group_column}' must be string or list of strings.")
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
                "split_column": split_column,
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
        fixed_ds = {
            _split: ds.filter(lambda x, k=_split: x[split_column] == k, **desc)
            for _split in nonfold_keys
        }
        if len(fold_keys) > 0:
            return tuple(
                DatasetDict({
                    "train": ds.filter(lambda x, k=_fold: x[split_column] != k and x[split_column] in fold_keys, **desc),
                    "validation": ds.filter(lambda x, k=_fold: x[split_column] == k, **desc),
                }
                | fixed_ds
                | {k: self.dataset[k] for k in self.dataset if k != key})
                for _fold in fold_keys
            )
        else:
            return DatasetDict(fixed_ds | {
                k: self.dataset[k] for k in self.dataset if k != key
            })

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
