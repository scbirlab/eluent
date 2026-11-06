"""Base splitting object."""

from typing import TYPE_CHECKING, Any
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
import json
from math import isfinite
from os import PathLike
from pathlib import Path

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset
    from providence.typing import DataLike
else:
    Dataset, DatasetDict, IterableDataset = Any, Any, Any
    DataLike = Any

from .decorators import process_splits
from .bin_packing import pack_bins
from .grouping import GROUPING_FUNCTIONS, annotate_split, annotate_split_invert, annotate_split_invert_fixed
from .utils import dataset_len, save_dataset

DEFAULT_BATCH_SIZE: int = 1024
DEFAULT_SEED: int = 42
SCHEMA_VERSION: int = 1
SCHEMA: dict[str, str | int] = {
    "schema_version": SCHEMA_VERSION,
    "info": "info.json",
    "partition_table": "partition.parquet",
    "resolved": "resolved.pv",
}
SCHEMA_FILENAME: str = "schema.json"


def _partition_by_bool_column(
    dataset: DatasetDict,
    fixed_columns: Iterable[str],
    fold_columns: Iterable[str] | None = None, 
    key: str = "train",
    fold_split_keys: Iterable[str] = ("train", "validation", "validation_partial")
) -> dict[DatasetDict]:
    from datasets import Dataset, DatasetDict

    ds = dataset[key]
    desc = {"desc": "Generating split datasets"} if isinstance(ds, Dataset) else {}
    other_splits = {k: d for k, d in dataset.items() if k != key}
    fixed_columns = list(fixed_columns)
    fold_columns = list(fold_columns or ())

    fixed_ds = {
        _split: ds.filter(lambda x, k=_split: x[k], **desc)
        for _split in fixed_columns
    }

    if fold_columns is None or len(fold_columns) == 0:
        return {"fold_x": DatasetDict(fixed_ds | other_splits)}

    return {
        fold: DatasetDict(
            {
                _split: ds.filter(
                    lambda x, k=f"{fold}|{_split}": x.get(k, False),
                    **desc,
                )
                for _split in fold_split_keys
            }
            | fixed_ds
            | other_splits
        )
        for fold in fold_columns
    }


@dataclass(frozen=True)
class SplitInfo:
    key: str
    counts: Mapping[str, Mapping[str, int]]
    fractions: Mapping[str, Mapping[str, float]]
    partition_columns: tuple[str]
    fixed_columns: tuple[str]
    fold_columns: tuple[str]
    group_columns: tuple[str]

    def to_config(self, path: str | PathLike):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def from_config(cls, path: str | PathLike | Mapping[str, Any]):
        if isinstance(path, Mapping):
            return cls(**path)
        elif isinstance(path, (str, PathLike)):
            path = Path(path)
            with open(path, "r") as f:
                d = json.load(f)
            return cls.from_config(d)
        else:
            raise ValueError(
                f"Path must be mapping or str or PathLike, but was {type(path)}: {path}"
            )


class SplitDataset:

    def __init__(
        self, 
        data: DataLike, 
        *args, 
        **kwargs
    ):
        from providence import DataResolver
        from datasets import Dataset, DatasetDict, IterableDataset
        if isinstance(data, DataResolver):
            resolved = data
        else:
            resolved = DataResolver(data)

        if isinstance(resolved.data, (Dataset, IterableDataset)):
            dataset = DatasetDict({"train": resolved.data})
        elif isinstance(resolved.data, DatasetDict):
            dataset = resolved.data
        else:
            raise ValueError(
                "Dataset must be Dataset, IterableDataset, or DatasetDict, "
                f"but was {type(resolved.data)}: {resolved.data}"
            )
        
        self._requested_data = data
        self.resolved = resolved
        self.dataset = dataset
        self.split_dataset = None
        self.split_info = None

    def save(
        self,
        path: str | PathLike,
        save_source_data: bool = False
    ):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        source_file_exists = (
            self.resolved.data_source.uri is not None 
            and Path(self.resolved.data_source.uri).exists()
        )
        reference_only = self.resolved.data_source.is_remote or source_file_exists
        self.resolved.save(
            path / SCHEMA["resolved"],
            save_source_data=save_source_data or not reference_only,
        )
        if self.split_info is not None:
            if isinstance(self.split_info, SplitInfo):
                self.split_info.to_config(path / SCHEMA["info"])
                partition_table = (
                    self.dataset[self.split_info.key]
                    .select_columns(list(self.split_info.partition_columns))
                )
                partition_table.to_parquet(path / SCHEMA["partition_table"])
            else:
                raise ValueError(f"Split info is not SplitInfo: {self.split_info}")
        else:
            raise ValueError(f"Split info is missing!")  # should never happen
        with open(path / SCHEMA_FILENAME, "w") as f:
            json.dump(SCHEMA, f, indent=4)
        return None

    def write_tables(
        self,
        path: str | PathLike,
        format: str = "parquet"
    ):
        from datasets import Dataset, DatasetDict, IterableDataset
        path = Path(path)
        format = format.lstrip(".")
        ds = self.split_dataset
        if isinstance(ds, DatasetDict):
            ds = {"fold_x": ds}
        elif isinstance(ds, (Dataset, IterableDataset)):
            ds = {"fold_x": DatasetDict({"train": ds})}
        elif isinstance(ds, Mapping):
            ds = dict(ds)
        else:
            # should never happen
            raise AttributeError(f"Dataset is not correct! {ds=}")
        for fold_key, fold in ds.items():
            fold_dir = path / str(fold_key)
            fold_dir.mkdir(parents=True, exist_ok=True)

            for key, split_ds in fold.items():
                split_nrows = dataset_len(split_ds)

                save_dataset(
                    split_ds, 
                    fold_dir / f"{key}.{format}",
                )
        if self.split_info is not None:
            if isinstance(self.split_info, SplitInfo):
                self.split_info.to_config(path / SCHEMA["info"])
            else:
                raise ValueError(f"Split info is not SplitInfo: {self.split_info}")

        return None            

    @classmethod
    def load(
        cls,
        path: str | PathLike,
        cache_dir: str
    ) -> "SplitDataset":
        from datasets import concatenate_datasets, DatasetDict
        from providence import DataResolver
        
        path = Path(path)
        with open(path / SCHEMA_FILENAME, "r") as f:
            schema = json.load(f)
        loaded_schema_version = schema["schema_version"]
        if loaded_schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Schema mismatch: laoded version {loaded_schema_version} "
                f"but package on version {SCHEMA_VERSION}"
            )
        resolved = DataResolver.load(
            path / schema["resolved"],
            cache_dir=cache_dir,
        )
        info = SplitInfo.from_config(path / schema["info"])
        partition_table = DataResolver(
            path / schema["partition_table"],
            cache_dir=cache_dir,
        ).data

        if isinstance(resolved.data, DatasetDict):
            source = source[info.key]

        resolved_len = dataset_len(resolved.data)
        partition_len = dataset_len(partition_table)
        if resolved_len != partition_len:
            raise ValueError(
                "Checkpoint partition length does not match source data: "
                f"{resolved_len=} != {partition_len=}"
            )

        missing = set(info.partition_columns) - set(partition_table.column_names)
        if missing:
            raise ValueError(
                f"Checkpoint is missing partition columns: {sorted(missing)}"
            )

        obj = cls(data=resolved)
        obj.split_info = info
        other_splits = {k: d for k,d in obj.dataset.items() if k != info.key}
        
        ds = concatenate_datasets([
            obj.dataset[info.key],
            partition_table,
        ], axis=1)
        obj.dataset = DatasetDict({
            **obj.dataset,
            info.key: ds,
        })
        split_dataset = _partition_by_bool_column(
            obj.dataset,
            fixed_columns=info.fixed_columns,
            fold_columns=info.fold_columns,
            key=info.key,
        )
        obj.split_dataset = split_dataset
        return obj

    def group(
        self,
        method: str | Callable,
        key: str = "train",
        group_column: str = "group",
        batch_size: int = DEFAULT_BATCH_SIZE,
        preprocess: Callable | None = None,
        **kwargs
    ) -> 'SplitDataset':
        """Run group annotation over the dataset, returning an annotated copy."""
        from datasets import Dataset, DatasetDict

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
            from .faiss import add_faiss_index
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
        from datasets import Dataset, DatasetDict

        ds = self.dataset[key]
        if not group_column or len(set(group_column)) != len(group_column):
            raise ValueError("Provide distinct grouping columns.")

        for column in group_column:
            if column not in ds.column_names:
                raise KeyError(f"Grouping column '{column}' is not in data: {ds.column_names}.")
        
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
        
        if len(fold_keys) > 0:
            split_to_groups, new_fold_keys = self._build_holdout_dict(
                split_keys=fold_keys,
                group_to_split=group_to_split,
                group_column=group_column,
                prefix=f"{key}:fold=",
            )

            cv_groups = tuple(
                {
                    group
                    for group, split_name in group_to_split[column].items()
                    if split_name in set(fold_keys)
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
        else:
            new_fold_keys = []
        
        return ds, nonfold_keys, new_fold_keys


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
    ) -> tuple[Dataset, Mapping, SplitInfo]:

        """Split a dataset based on grouping.

        ``group_column`` may be a column name or a sequence of independent
        grouping axes. Multiple axes use intersections: rows with conflicting
        split assignments are excluded. 
        Fold combinations are enumerated in Cartesian-product order; 
        each returned evaluation is group-disjoint
        on every axis. 
        Fractions apply per axis, before intersection.

        Examples
        ========
        Construct every combination of four groups on three axes.

        >>> from itertools import combinations, product
        >>> from datasets import Dataset, DatasetDict
        >>> axes = ("scaffold", "taxon", "target")
        >>> rows = list(product(range(4), repeat=3))
        >>> data = Dataset.from_dict({
        ...     "id": list(range(len(rows))),
        ...     **{
        ...         axis: [row[i] for row in rows]
        ...         for i, axis in enumerate(axes)
        ...     },
        ... })

        Ordinary splits retain intersections and exclude mixed assignments.

        >>> _, parts, info = SplitDataset(data).split(
        ...     group_column=axes,
        ...     splits={"train": 0.5, "test": 0.5},
        ... )
        >>> sorted(parts["fold_x"])
        ['test', 'train']
        >>> parts = parts["fold_x"]
        >>> len(parts["train"]), len(parts["test"])
        (8, 8)
        >>> all(
        ...     set(parts["train"][axis]).isdisjoint(parts["test"][axis])
        ...     for axis in axes
        ... )
        True

        Three CV folds across three axes produce 27 evaluations.

        >>> _, folds, info = SplitDataset(data).split(
        ...     group_column=axes,
        ...     splits={
        ...         "train:fold=0": 0.25,
        ...         "train:fold=1": 0.25,
        ...         "train:fold=2": 0.25,
        ...         "test": 0.25,
        ...     },
        ... )
        >>> folds = list(folds.values())
        >>> len(folds)
        27
        >>> all(
        ...     set(fold) == {"train", "validation", "validation_partial", "test"}
        ...     for fold in folds
        ... )
        True

        The Cartesian fixture populates every returned partition.

        >>> all(
        ...     (
        ...         len(fold["train"]),
        ...         len(fold["validation"]),
        ...         len(fold["validation_partial"]),
        ...         len(fold["test"]),
        ...     ) == (8, 1, 18, 1)
        ...     for fold in folds
        ... )
        True

        Full train, validation and test partitions exclude groups on every axis.

        >>> all(
        ...     set(fold[left][axis]).isdisjoint(fold[right][axis])
        ...     for fold in folds
        ...     for left, right in combinations(("train", "validation", "test"), 2)
        ...     for axis in axes
        ... )
        True

        Fixed-test groups are excluded from partial validation as well.

        >>> all(
        ...     set(fold["test"][axis]).isdisjoint(fold["validation_partial"][axis])
        ...     for fold in folds
        ...     for axis in axes
        ... )
        True

        All returned partitions are disjoint by observation.

        >>> all(
        ...     set(fold[left]["id"]).isdisjoint(fold[right]["id"])
        ...     for fold in folds
        ...     for left, right in combinations(fold, 2)
        ... )
        True

        The fixed test set stays identical across evaluations.

        >>> all(
        ...     set(fold["test"]["id"]) == set(folds[0]["test"]["id"])
        ...     for fold in folds
        ... )
        True

        Every CV-eligible observation receives full validation exactly once.

        >>> from collections import Counter
        >>> validation_counts = Counter(
        ...     row_id
        ...     for fold in folds
        ...     for row_id in fold["validation"]["id"]
        ... )
        >>> len(validation_counts), set(validation_counts.values())
        (27, {1})

        Unrelated DatasetDict entries pass through.

        >>> source = DatasetDict({"train": data, "external": data.select([0])})
        >>> _, parts, _ = SplitDataset(source).split(
        ...     group_column=axes,
        ...     splits={"train": 0.5, "test": 0.5},
        ... )
        >>> list(parts["fold_x"]["external"]["id"])
        [0]

        Duplicate grouping axes are rejected.

        >>> SplitDataset(data).split(
        ...     group_column=("scaffold", "scaffold"),
        ...     splits={"train": 0.5, "test": 0.5},
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Provide distinct grouping columns.

        """
        from datasets import DatasetDict
        ds = self.dataset[key]
        if isinstance(group_column, str):
            group_column = (group_column,)
        elif isinstance(group_column, Iterable):
            group_column = tuple(group_column)
        else:
            raise ValueError(f"Grouping column '{group_column}' must be string or list of strings.")
        missing_columns = set(group_column) - set(ds.column_names)
        if len(missing_columns) > 0: 
            raise KeyError(f"Grouping columns [{missing_columns}] is not in data: {ds.column_names}")

        ds, fixed_columns, fold_columns = self._split_multiway(
            key=key,
            group_column=group_column,
            deterministic=deterministic,
            seed=seed,
            batch_size=batch_size,
            splits=splits,
            **kwargs,
        )
        split_ds = _partition_by_bool_column(
            DatasetDict({**self.dataset, key: ds}),
            fixed_columns=fixed_columns,
            fold_columns=fold_columns,
            key=key,
        )
        partition_columns = [
            f"{col}|{split_key}" 
            for col, datadict in split_ds.items()
            for split_key in datadict
            if f"{col}|{split_key}" in ds.column_names
        ] + fixed_columns
        split_total = dataset_len(ds)
        split_counter = {
            fold_key: {key: dataset_len(d) for key, d in _ds.items()}
            for fold_key, _ds in split_ds.items()
        }
        split_fracs = {
            fold_key: {key: d / split_total for key, d in _ds.items()}
            for fold_key, _ds in split_counter.items()
        }
        split_info = SplitInfo(
            key=key,
            counts=split_counter,
            fractions=split_fracs,
            partition_columns=tuple(partition_columns),
            fixed_columns=tuple(fixed_columns),
            fold_columns=tuple(fold_columns),
            group_columns=tuple(group_column),
        )
        return ds, split_ds, split_info


    @process_splits
    def group_and_split(
        self,
        method: str | Callable | None = None,
        key: str = "train",
        group_column: str | Iterable[str] = "group",
        split_column: str = "split",
        deterministic: bool = True,
        seed: int = DEFAULT_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        splits: Mapping[str, float] | None = None,
        preprocess: Callable | None = None,
        **kwargs
    ):
        """Annotate groups when requested, split the data, and return self.

        Examples
        ========
        Construct every combination of four groups on three axes.

        >>> from itertools import product
        >>> axes = ("scaffold", "taxon", "target")
        >>> rows = list(product(range(4), repeat=3))
        >>> data = Dataset.from_dict({
        ...     "id": list(range(len(rows))),
        ...     **{
        ...         axis: [row[i] for row in rows]
        ...         for i, axis in enumerate(axes)
        ...     },
        ... })

        Existing grouping columns can be split without generating new labels.
        Explicit test fractions leave no inferred validation partition.

        >>> splitter = SplitDataset(data)
        >>> result = splitter.group_and_split(
        ...     method=None,
        ...     group_column=axes,
        ...     train=0.5,
        ...     test=0.5,
        ... )
        >>> result is splitter
        True
        >>> parts = result.split_dataset["fold_x"]
        >>> sorted(parts)
        ['test', 'train']
        >>> len(parts["train"]), len(parts["test"])
        (8, 8)
        >>> all(
        ...     set(parts["train"][axis]).isdisjoint(parts["test"][axis])
        ...     for axis in axes
        ... )
        True

        Without a test fraction, the remainder becomes validation.

        >>> result = SplitDataset(data).group_and_split(
        ...     method=None,
        ...     group_column=axes,
        ...     train=0.5,
        ... )
        >>> parts = result.split_dataset["fold_x"]
        >>> sorted(parts)
        ['train', 'validation']
        >>> len(parts["train"]), len(parts["validation"])
        (8, 8)

        Three folds on each of three axes produce 27 evaluations.
        The remaining quarter of each axis is reserved for fixed test.

        >>> result = SplitDataset(data).group_and_split(
        ...     method=None,
        ...     group_column=axes,
        ...     train=0.75,
        ...     test=0.25,
        ...     kfolds=3,
        ... )
        >>> folds = list(result.split_dataset.values())
        >>> len(folds)
        27
        >>> all(
        ...     set(fold) == {"train", "validation", "validation_partial", "test"}
        ...     for fold in folds
        ... )
        True
        >>> all(
        ...     (
        ...         len(fold["train"]),
        ...         len(fold["validation"]),
        ...         len(fold["validation_partial"]),
        ...         len(fold["test"]),
        ...     ) == (8, 1, 18, 1)
        ...     for fold in folds
        ... )
        True

        Generating groups and then splitting still works for a single axis.

        >>> result = SplitDataset(data).group_and_split(
        ...     method="random",
        ...     train=0.5,
        ...     test=0.5,
        ...     seed=42,
        ... )
        >>> parts = result.split_dataset["fold_x"]
        >>> sorted(parts)
        ['test', 'train']
        >>> all("group" in part.column_names for part in parts.values())
        True
        >>> train_ids = set(parts["train"]["id"])
        >>> test_ids = set(parts["test"]["id"])
        >>> train_ids.isdisjoint(test_ids)
        True
        >>> train_ids | test_ids == set(data["id"])
        True
        
        """
        from datasets import DatasetDict
        if method is not None:
            self.dataset = self.group(
                key=key,
                group_column=group_column,
                method=method,
                batch_size=batch_size,
                preprocess=preprocess,
                **kwargs,
            )
        dataset, split_dataset, info = self.split(
            key=key,
            group_column=group_column,
            split_column=split_column,
            deterministic=deterministic,
            seed=seed,
            batch_size=batch_size,
            splits=splits,
            **kwargs,
        )
        self.dataset = DatasetDict({
            **self.dataset,
            key: dataset,
        })
        self.split_dataset = split_dataset
        self.split_info = info
        return self
