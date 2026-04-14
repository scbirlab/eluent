"""Generating approaximate spectral splits on chemical data using FAISS index."""

from typing import Dict, Iterable, Optional, Mapping, Union

from datasets import Dataset, IterableDataset
import faiss
import numpy as np
from schemist.features import calculate_feature
from tqdm.auto import tqdm, trange

from .bin_packing import pack_bins
from .decorators import process_splits
from .disjoint_set import NumpyDisjointSet
from .utils import annotate_split, dataset_len


def _morgan_fingerprint(
    x: Mapping[str, Iterable],
    structure_column: str, 
    input_representation: str = 'smiles',
    fingerprint_column: str = "fp"
) -> np.ndarray:
    """Compute Morgan fingerprints and pack bits into uint8 array.

    Examples
    ========
    >>> batch = {'smiles': ['CCO', 'CCC']}
    >>> out = _morgan_fingerprint(batch, structure_column='smiles')
    >>> 'fp' in out
    True
    >>> isinstance(out['fp'], np.ndarray)
    True
    >>> out['fp'].shape == (2, 2048 // 8)
    True

    """
    fingerprints, _ = calculate_feature(
        strings=x[structure_column],
        feature_type="fp",
        return_dataframe=False,
        on_bits=False,
    )
    x[fingerprint_column] = np.packbits(fingerprints.astype(bool), axis=-1)
    return x


def _annotate_component(
    x: Mapping[str, str], 
    idx: int,
    disjoint_set: NumpyDisjointSet,
    component_column: str = "faiss_component"
) -> Dict[str, ...]:
    """Tag each entry with its graph component.

    """
    x[component_column] = disjoint_set.find(idx)
    return x


def annotate_faiss_components(
    ds: Union[Dataset, IterableDataset],
    structure_column: str,
    input_representation: str = "smiles",
    fingerprint_column: str = "_faiss_fp_",
    n_neighbors: int = 10,
    batch_size: int = 1024,
    component_column: str = "faiss_component",
    gpu: bool = False
) -> Union[Dataset, IterableDataset]:
    """Compute Morgan fingerprints, build FAISS k-NN graph, annotate components.
    
    """
    faiss_index = faiss.IndexBinaryFlat(2048)  # Hamming/Jaccard on 2048-bit vectors
    ds = (
        ds
        .map(
            _morgan_fingerprint,
            fn_kwargs={
                "structure_column": structure_column,
                "input_representation": input_representation,
                "fingerprint_column": fingerprint_column,
            },
            batched=True,
            batch_size=batch_size,
            desc="Annotating fingerprint",
        )
    )
    ds = ds.add_faiss_index(
        column=fingerprint_column, 
        index_name=f"{fingerprint_column}_idx",
        custom_index=faiss_index,
        device=0 if gpu else None,
        dtype=np.uint8,
    )

    N = dataset_len(ds)
    djs = NumpyDisjointSet(N)

    iterator = tqdm(
        ds.iter(batch_size=batch_size),
        desc=f"Finding {n_neighbors} nearest neighbors per molecule",
        total=np.ceil(N / batch_size).astype(int),
    )
    for i, batch in enumerate(iterator):
        start = i * batch_size
        batch_fp = np.asarray(batch[fingerprint_column]).astype(np.uint8)
        scores, indices = ds.search_batch(
            f"{fingerprint_column}_idx", 
            batch_fp, 
            k=n_neighbors,
        )
        for row_of_batch, neighbor_id_list in enumerate(indices):
            row_of_dataset = start + row_of_batch
            # skip neighbor_id_list[0] because it is itself
            for neighbor_id in neighbor_id_list[1:]:
                djs.merge(row_of_dataset, int(neighbor_id))

    # path-compression pass
    for i in trange(N, desc="Compressing search path"):
        djs.parent[i] = djs.find(i)

    # prevents pickling error in .map()
    ds.drop_index(f"{fingerprint_column}_idx") 

    if isinstance(ds, Dataset):
        desc = {"desc": "Annotating components"}
    else:
        desc = {}
    return N, ds.map(
        _annotate_component,
        fn_kwargs={
            "disjoint_set": djs,
            "component_column": component_column,
        },
        with_indices=True, 
        batched=False,
        **desc,
    )


@process_splits
def faiss_split(
    ds: Union[Dataset, IterableDataset],
    structure_column: str,
    input_representation: str = "smiles",
    fingerprint_column: str = "_faiss_fp_",
    component_column: str = "faiss_component",
    n_neighbors: int = 10,
    batch_size: int = 1024,
    splits: Optional[Mapping[str, float]] = None,
    deterministic: bool = True,
    seed: int = 42,
    cache: str = "cache",
    gpu: bool = False
):
    """Approximate spectral split using FAISS and union-find.

    For an empty dataset, returns empty splits.

    """
    N, ds = annotate_faiss_components(
        ds, 
        structure_column=structure_column,
        input_representation=input_representation,
        fingerprint_column=fingerprint_column,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
        component_column=component_column,
        gpu=gpu,

    )
    component_to_split = pack_bins(
        ds,
        group_column="faiss_component",
        splits=splits,
        batch_size=batch_size,
        num_rows=N,
        deterministic=deterministic,
        seed=seed,
    )

    if isinstance(ds, Dataset):
        desc = {"desc": "Annotating splits"}
    else:
        desc = {}
    ds = (
        ds
        .map(
            annotate_split,
            fn_kwargs={
                "key_to_split": component_to_split,
                "key_column": "faiss_component",
                "split_column": "split",
            },
            batched=True,
            batch_size=batch_size,
            **desc,
        )
    )
    if isinstance(ds, Dataset):
        desc = {"desc": "Generating split datasets"}
    else:
        desc = {}
    return ds, {
        key: ds.filter(lambda x: x["split"] == key, **desc)
        for key in splits
    }
