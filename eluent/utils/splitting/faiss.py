"""Generating approaximate spectral splits on chemical data using FAISS index."""

from typing import Dict, Iterable, Optional, Mapping, Union

from datasets import Dataset, IterableDataset
import faiss
import numpy as np
from tqdm.auto import tqdm, trange

from .disjoint_set import NumpyDisjointSet
from .grouping import _morgan_fingerprint
from .utils import dataset_len


def add_faiss_index(
    ds: Union[Dataset, IterableDataset],
    structure_column: str,
    n_neighbors: int = 10,
    input_representation: str = "smiles",
    fingerprint_column: str = "_faiss_fp_",
    batch_size: int = 1024,
    gpu: bool = False,
    **kwargs
) -> Union[Dataset, IterableDataset]:
    """Compute Morgan fingerprints, build FAISS k-NN graph, annotate components.
    
    """
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
    ds = ds.add_faiss_index(
        column=fingerprint_column, 
        index_name=f"{fingerprint_column}_idx",
        custom_index=faiss.IndexBinaryFlat(2048),  # Hamming/Jaccard on 2048-bit vectors
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
    return ds, {"disjoint_set": djs}
