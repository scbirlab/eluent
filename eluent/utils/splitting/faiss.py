"""Generating approaximate spectral splits on chemical data using FAISS index."""

from typing import Dict, Iterable, Optional, Mapping, Union

from datasets import Dataset, IterableDataset
import faiss
import numpy as np
from tqdm.auto import tqdm, trange

from .disjoint_set import NumpyDisjointSet
from .grouping import _morgan_fingerprint
from .utils import dataset_len

FAISS_COMPONENT_CACHE_VERSION = 1


def add_faiss_index(
    ds: Union[Dataset, IterableDataset],
    structure_column: str,
    n_neighbors: int = 10,
    input_representation: str = "smiles",
    fingerprint_column: str = "_faiss_fp_",
    batch_size: int = 1024,
    gpu: bool = False,
    cache: str | None = None,
    **kwargs
) -> Union[Dataset, IterableDataset]:
    """Compute Morgan fingerprints, build FAISS k-NN graph, annotate components.
    
    """
    import os

    from filelock import FileLock
    from providence.package_data import configure_hf_cache

    from ..datasets_io import hasher, _lock_path

    cache, *_ = configure_hf_cache(cache)
    ds_fingerprint = ds._fingerprint
    cache_key = "::".join([
        f"faiss-components-v{FAISS_COMPONENT_CACHE_VERSION}",
        ds_fingerprint,
        structure_column,
        input_representation,
        str(n_neighbors),
        str(gpu),
    ])
    ds = (
        ds
        .map(
            _morgan_fingerprint,
            fn_kwargs={
                "structure_column": structure_column,
                "input_representation": input_representation,
                "column": fingerprint_column,
            },
            batched=True,
            batch_size=batch_size,
            desc="Annotating fingerprint",
        )
    )
    component_cache_dir = os.path.join(
        cache,
        "eluent",
        "faiss-components",
    )
    os.makedirs(component_cache_dir, exist_ok=True)

    component_cache_file = os.path.join(
        component_cache_dir,
        f"{hasher(cache_key)}.npy",
    )

    lockfile = _lock_path(
        key=cache_key,
        cache_dir=cache,
    )
    if os.path.exists(component_cache_file):
        component_labels = np.load(
            component_cache_file,
            allow_pickle=False,
        )
        return ds, {
            "component_labels": component_labels,
        }
    with FileLock(lockfile, timeout=60. * 60.):
        if os.path.exists(component_cache_file):
            component_labels = np.load(
                component_cache_file,
                allow_pickle=False,
            )
            return ds, {
                "component_labels": component_labels,
            }

        if os.path.exists(component_cache_file):
            component_labels = np.load(
                component_cache_file,
                allow_pickle=False,
            )
            return ds, {
                "component_labels": component_labels,
            }

        ds_fp = ds.map(
            _morgan_fingerprint,
            fn_kwargs={
                "structure_column": structure_column,
                "input_representation": input_representation,
                "column": fingerprint_column,
            },
            batched=True,
            batch_size=batch_size,
            desc="Annotating fingerprint",
        )

        ds_fp = ds_fp.add_faiss_index(
            column=fingerprint_column,
            index_name=f"{fingerprint_column}_idx",
            custom_index=faiss.IndexBinaryFlat(2048),
            device=0 if gpu else None,
            dtype=np.uint8,
        )

        N = dataset_len(ds_fp)
        djs = NumpyDisjointSet(N)

        iterator = tqdm(
            ds_fp.iter(batch_size=batch_size),
            desc=f"Finding {n_neighbors} nearest neighbors per molecule",
            total=np.ceil(N / batch_size).astype(int),
        )
        for i, batch in enumerate(iterator):
            start = i * batch_size
            batch_fp = np.asarray(
                batch[fingerprint_column]
            ).astype(np.uint8)

            _, indices = ds_fp.search_batch(
                f"{fingerprint_column}_idx",
                batch_fp,
                k=n_neighbors,
            )

            for row_of_batch, neighbor_id_list in enumerate(indices):
                row_of_dataset = start + row_of_batch
                for neighbor_id in neighbor_id_list[1:]:
                    djs.merge(
                        row_of_dataset,
                        int(neighbor_id),
                    )

        component_labels = np.empty(N, dtype=np.int32)
        for i in trange(N, desc="Compressing search path"):
            component_labels[i] = djs.find(i)

        ds_fp.drop_index(
            f"{fingerprint_column}_idx"
        )
        tmp_file = (
            f"{component_cache_file}."
            f"{os.getpid()}.tmp.npy"
        )
        np.save(
            tmp_file,
            component_labels,
            allow_pickle=False,
        )
        os.replace(
            tmp_file,
            component_cache_file,
        )
        return ds, {
            "component_labels": component_labels,
        }