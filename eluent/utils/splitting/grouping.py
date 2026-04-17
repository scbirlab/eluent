"""Generating splits on chemical scaffold."""

from typing import Callable, Dict, Iterable, Optional, Mapping, Union
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
from .disjoint_set import NumpyDisjointSet
from .utils import annotate_split

GROUPING_FUNCTIONS: Dict[str, Callable] = {}

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
    ) -> Dict[str, ...]:
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
) -> Dict[str, Iterable]:
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
    >>> out['fingerprint_column'].shape == (2, 2048 // 8)
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
    disjoint_set: NumpyDisjointSet,
    **kwargs
) -> Dict[str, ...]:
    """Tag each entry with its graph component.

    """
    return [disjoint_set.find(i) for i in indices]


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
