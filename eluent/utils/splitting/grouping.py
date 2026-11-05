"""Generating split groups."""

from collections.abc import Callable, Iterable, Mapping
from functools import partial, wraps
import hashlib
import struct

from carabiner.decorators import decorator_with_params
from datasets import Dataset, IterableDataset
import numpy as np
from schemist.converting import convert_string_representation
from schemist.features import calculate_feature

from .bin_packing import pack_bins
from .decorators import process_splits

GROUPING_FUNCTIONS: dict[str, Callable] = {}


@decorator_with_params
def _add_to_grouping_registry(fn: Callable, key: str) -> Callable:
    GROUPING_FUNCTIONS[key] = fn
    return fn


# @decorator_with_params
def _assign_and_return(
    fn: Callable
) -> Callable:

    @wraps(fn)
    def _fn(
        x: Mapping[str, Iterable],
        *args,
        column: str = "group",
        **kwargs
    ) -> dict[str, ...]:
        values = fn(x, *args, **kwargs)
        x[column] = values
        return x
    return _fn


@_assign_and_return
def tag_top(
    x: Mapping[str, Iterable],
    key: str,
    cutoff: float,
    delta: float = .01,
    reverse: bool = False
) -> dict[str, Iterable]:
    out = []
    sign = -1 if reverse else 1
    for v in x[key]:
        if v >= cutoff + delta:
            out.append(sign)
        elif v <= cutoff - delta:
            out.append(-1 * sign)
        else:
            out.append(0)
    return out


@_assign_and_return
def _morgan_fingerprint(
    x: Mapping[str, Iterable],
    structure_column: str, 
    input_representation: str = "smiles",
    **kwargs
) -> np.ndarray:
    """Compute Morgan fingerprints and pack bits into uint8 array.

    Examples
    ========
    >>> batch = {'smiles': ['CCO', 'CCC']}
    >>> out = _morgan_fingerprint(batch, structure_column='smiles')
    >>> out['group'].shape == (2, 2048 // 8)
    True

    """
    fingerprints, _ = calculate_feature(
        strings=x[structure_column],
        feature_type="fp",
        return_dataframe=False,
        on_bits=False,
    )
    return np.packbits(fingerprints.astype(bool), axis=-1)


@_add_to_grouping_registry(key="faiss")
@_assign_and_return
def _faiss_component(
    x: Mapping[str, str], 
    indices: Iterable[int], 
    component_labels: np.ndarray,
    **kwargs
) -> dict[str, ...]:
    """Tag each entry with its graph component.

    """
    return component_labels[np.asarray(indices)].tolist()


@_add_to_grouping_registry(key="random")
@_assign_and_return
def _random(
    x: Mapping[str, Iterable], 
    indices: Iterable[int], 
    seed: int = 42,
    **kwargs
):
    """Produce a reproducible pseudo-unique uint64 per row."""
    return [
        struct.unpack(
            '>Q', 
            hashlib.sha256(
                struct.pack('>QQ', seed, idx)
            ).digest()[:8])[0] & 0x7FFFFFFFFFFFFFFF
        for idx in indices
    ]

@_add_to_grouping_registry(key="scaffold")
@_assign_and_return
def _chemical_scaffold(
    x: Mapping[str, Iterable],
    indices: Iterable[int], 
    structure_column: str, 
    input_representation: str = "smiles",
    scaffold_classifier: str = "scaffold",
    group_column: str = "scaffold_group",
    **kwargs
):
    """Annotate a chemical structure with its scaffold.

    """

    converted = convert_string_representation(
        strings=x[structure_column],
        input_representation=input_representation,
        output_representation=scaffold_classifier,
    )
    if len(x[structure_column]) > 1 and not isinstance(x[structure_column], str):
        converted = list(converted)
    else:
        converted = [converted]
    return converted


def annotate_split(
    x: Mapping[str, ...], 
    key_to_split: Mapping[str, str],
    group_column: str | Iterable[str] = "group",
    split_column: str = "split"
) -> dict[str, ...]:
    """Tag each entry with its split label.

    Examples
    ========
    >>> comp_map = {"A": "train", "B": "test"}
    >>> out = annotate_split({"col": ["B", "A"]}, key_to_split=comp_map, group_column="col", split_column="split")
    >>> out["split"]
    ['test', 'train']
    >>> comp_map = {("A", "A"): "train", ("A", "B"): "test"}
    >>> out = annotate_split({"col": ["A", "A"], "col2": ["B", "A"]}, key_to_split=comp_map, group_column=("col", "col2"), split_column="split")
    >>> out["split"]
    ['test', 'train']


    """
    columns = (group_column,) if isinstance(group_column, str) else tuple(group_column)
    if not columns:
        raise ValueError("Provide at least one grouping column.")

    if len(columns) == 1:
        keys = list(x[columns[0]])
    else:
        keys = list(zip(*(x[column] for column in columns), strict=True))

    missing = set(keys).difference(key_to_split)
    if missing:
        raise ValueError(f"Groups not found in split mapping: {missing}")

    x[split_column] = [key_to_split[key] for key in keys]
    return x


def annotate_split_invert(
    x: Mapping[str, ...], 
    split_to_group: Mapping[str, Iterable[Iterable]],
    group_column: str | Iterable[str] = "group",
    *,
    cv_groups
) -> dict[str, ...]:
    """Annotate CV membership for each held-out fold combination.

    Each holdout and ``cv_groups`` contain one set per grouping axis, in
    ``group_column`` order. Rows outside the CV pool have all flags false.

    Examples
    ========
        Examples
    ========
    Each tuple contains one group set per axis.

    >>> batch = {
    ...     "a": ["A1", "A0", "A0", "AT", "AT"],
    ...     "b": ["B1", "B0", "B1", "B1", "BT"],
    ... }
    >>> holdouts = {"fold0": ({"A0"}, {"B0"})}
    >>> out = annotate_split_invert(
    ...     batch,
    ...     split_to_group=holdouts,
    ...     group_column=("a", "b"),
    ...     cv_groups=({"A0", "A1"}, {"B0", "B1"}),
    ... )
    >>> out["fold0|train"]
    [True, False, False, False, False]
    >>> out["fold0|validation"]
    [False, True, False, False, False]
    >>> out["fold0|validation_partial"]
    [False, False, True, False, False]

    Single-axis splitting has no partial holdouts.

    >>> out = annotate_split_invert(
    ...     {"a": ["A0", "A1", "AT"]},
    ...     split_to_group={"fold0": ({"A0"},)},
    ...     group_column="a",
    ...     cv_groups=({"A0", "A1"},),
    ... )
    >>> out["fold0|train"]
    [False, True, False]
    >>> out["fold0|validation"]
    [True, False, False]
    >>> out["fold0|validation_partial"]
    [False, False, False]

    """
    if isinstance(group_column, str):
        group_column = (group_column,)

    rows = list(zip(*(x[column] for column in group_column)))
    eligible = [
        all(
            value in allowed 
            for value, allowed in zip(row, cv_groups)
        )
        for row in rows
    ]

    for fold_name, holdouts in split_to_group.items():
        train = []
        validation = []
        validation_partial = []

        for row, in_cv in zip(rows, eligible):
            held = tuple(
                value in holdout
                for value, holdout in zip(row, holdouts)
            )
            train.append(in_cv and not any(held))
            validation.append(in_cv and all(held))
            validation_partial.append(in_cv and any(held) and not all(held))

        x[f"{fold_name}|train"] = train
        x[f"{fold_name}|validation"] = validation
        x[f"{fold_name}|validation_partial"] = validation_partial

    return x


def annotate_split_invert_fixed(
    x: Mapping[str, ...], 
    split_to_group: Mapping[str, Iterable[Iterable]],
    group_column: str | Iterable[str] = "group"
) -> dict[str, ...]:
    """Annotate fixed-split membership by requiring a match on every axis.

    Each fixed split contains one set per grouping axis, in
    ``group_column`` order.

    Examples
    ========
    Fixed splits require membership on every axis.

    >>> out = annotate_split_invert_fixed(
    ...     {"a": ["AT", "AT", "A1", "A1"], "b": ["BT", "B1", "BT", "B1"]},
    ...     split_to_group={"test": ({"AT"}, {"BT"})},
    ...     group_column=("a", "b"),
    ... )
    >>> out["test"]
    [True, False, False, False]

    Single-axis fixed splits use a tuple containing one set.

    >>> out = annotate_split_invert_fixed(
    ...     {"a": ["AT", "A1"]},
    ...     split_to_group={"test": ({"AT"},)},
    ...     group_column="a",
    ... )
    >>> out["test"]
    [True, False]

    """
    if isinstance(group_column, str):
        group_column = (group_column,)
    elif isinstance(group_column, (list, tuple)):
        group_column = tuple(group_column)
    else:
        raise ValueError(
            "Group column must be iterable or string, "
            f"but was {type(group_column)}: {group_column=}"
        )

    x_transposed = list(zip(*(x[g] for g in group_column)))
    for col, holdouts in split_to_group.items():
        x[col] = [
            all(d in h for d, h in zip(row, holdouts))
            for row in x_transposed
        ]
    return x
