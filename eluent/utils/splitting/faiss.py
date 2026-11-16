"""Generating approximate spectral splits on chemical data using FAISS index."""

from typing import Dict, Iterable, Optional, Mapping, Tuple, Union

from datasets import Dataset, IterableDataset
import faiss
import numpy as np
from tqdm.auto import tqdm, trange

from .disjoint_set import NumpyDisjointSet
from .grouping import _morgan_fingerprint
from .utils import dataset_len

FAISS_COMPONENT_CACHE_VERSION = 2


def _infer_fp_params(
    ds: Union[Dataset, IterableDataset],
    fingerprint_column: str,
) -> Tuple[int, np.dtype]:
    """Return ``(vector_dimension, element_numpy_dtype)`` for a fingerprint column.

    Uses ``ds.features`` for dtype (reliable Arrow schema); peeks at the first
    element for dimension when the feature length is unspecified.

    Dtype rejection for unsupported types (e.g. non-uint8 integers) is handled
    downstream in ``_build_faiss_index``.  Ragged vectors are not fully validated
    here; FAISS will raise on a mismatched dimension during index construction.

    Raises ``ValueError`` for unsupported feature types (nested sequences) and
    zero-length vectors.

    Examples
    ========
    >>> import numpy as np
    >>> from datasets import Dataset, Sequence, Value
    >>> ds = Dataset.from_dict({"fp": [list(np.ones(32, dtype=np.uint8))]})
    >>> _infer_fp_params(ds, "fp")
    (32, dtype('uint8'))

    """
    from datasets import Sequence, Value

    features = getattr(ds, "features", None)
    elem_dtype: Optional[np.dtype] = None
    dim: Optional[int] = None

    if features is not None and fingerprint_column in features:
        feature = features[fingerprint_column]
        if not isinstance(feature, Sequence):
            raise ValueError(
                f"Column '{fingerprint_column}' has feature type "
                f"{type(feature).__name__}; expected a Sequence of scalars."
            )
        elem_feature = feature.feature
        if not isinstance(elem_feature, Value):
            raise TypeError(
                f"Column '{fingerprint_column}' contains nested sequences; "
                "expected a flat vector of scalars."
            )
        elem_dtype = np.dtype(elem_feature.dtype)
        if feature.length > 0:
            dim = feature.length

    if dim is None or elem_dtype is None:
        # Schema absent or variable-length: peek at first element.
        first_batch = next(iter(ds.iter(batch_size=1)))
        first_vec = np.asarray(first_batch[fingerprint_column][0])
        if first_vec.ndim != 1 or first_vec.size == 0:
            raise ValueError(
                f"Column '{fingerprint_column}' does not contain flat, non-empty "
                "vectors."
            )
        if dim is None:
            dim = int(first_vec.size)
        # Only use peeked dtype as fallback; schema dtype is preferred.
        if elem_dtype is None:
            elem_dtype = first_vec.dtype

    if dim == 0:
        raise ValueError(
            f"Column '{fingerprint_column}' contains zero-dimensional vectors."
        )

    return dim, elem_dtype


def _build_faiss_index(
    d: int,
    elem_dtype: np.dtype,
    gpu: bool = False,
) -> Tuple:
    """Return ``(custom_index, fp_dtype, gpu_resources)`` appropriate for the vector type.

    - ``uint8`` → ``IndexBinaryFlat(d * 8)`` (Hamming over packed bits).
    - float → ``IndexFlatL2(d)`` (L2, vectors cast to ``float32``).
    - Other dtypes raise ``ValueError``.

    GPU is applied to float indexes only; a warning is emitted for binary
    indexes since ``IndexBinaryFlat`` is CPU-only.

    ``gpu_resources`` is a ``faiss.StandardGpuResources`` instance when GPU is
    used, otherwise ``None``.  Callers must keep it alive for as long as the
    index is in use.

    Examples
    ========
    >>> import numpy as np
    >>> idx, dt, _ = _build_faiss_index(32, np.dtype("uint8"))
    >>> isinstance(idx, faiss.IndexBinaryFlat), dt == np.uint8
    (True, True)
    >>> idx, dt, _ = _build_faiss_index(128, np.dtype("float32"))
    >>> isinstance(idx, faiss.IndexFlatL2), dt == np.float32
    (True, True)

    """
    gpu_resources = None
    if elem_dtype == np.dtype("uint8"):
        custom_index = faiss.IndexBinaryFlat(d * 8)
        fp_dtype = np.uint8
        if gpu:
            import warnings
            warnings.warn(
                "GPU is not supported for binary FAISS indexes; running on CPU.",
                stacklevel=3,
            )
    elif np.issubdtype(elem_dtype, np.floating):
        custom_index = faiss.IndexFlatL2(d)
        fp_dtype = np.float32
        if gpu:
            gpu_resources = faiss.StandardGpuResources()
            custom_index = faiss.index_cpu_to_gpu(gpu_resources, 0, custom_index)
    else:
        raise ValueError(
            f"Unsupported fingerprint element dtype '{elem_dtype}'. "
            "Use uint8 for packed binary fingerprints (Morgan) or a float "
            "type for continuous embeddings."
        )
    return custom_index, fp_dtype, gpu_resources


def add_faiss_index(
    ds: Union[Dataset, IterableDataset],
    structure_column: str | None = None,
    n_neighbors: int = 10,
    input_representation: str = "smiles",
    fingerprint_column: str = "_faiss_fp_",
    batch_size: int = 1024,
    gpu: bool = False,
    cache: str | None = None,
    **kwargs
) -> Union[Dataset, IterableDataset]:
    """Compute Morgan fingerprints, build FAISS k-NN graph, annotate components.

    If ``fingerprint_column`` already exists in ``ds``, it is used directly and
    ``structure_column`` is ignored.  Otherwise Morgan fingerprints are computed
    from ``structure_column`` and stored under ``fingerprint_column``.

    The FAISS index type is inferred from the Arrow schema of ``fingerprint_column``:
    ``uint8`` sequences use ``IndexBinaryFlat`` (Hamming distance); float sequences
    use ``IndexFlatL2`` (Euclidean, cast to ``float32``).
    """
    import os

    from filelock import FileLock
    from providence.package_data import configure_hf_cache
    from providence.io import hasher, _lock_path

    from .utils import check_new_columns

    use_precomputed = fingerprint_column in ds.column_names
    if not use_precomputed:
        if structure_column is None:
            raise ValueError(
                f"fingerprint_column '{fingerprint_column}' is not in the dataset "
                "and structure_column was not provided.  Pass structure_column to "
                "compute Morgan fingerprints, or supply a pre-computed "
                f"'{fingerprint_column}' column."
            )
        check_new_columns(ds.column_names, [fingerprint_column], "FAISS fingerprint")

    cache, *_ = configure_hf_cache(cache)
    ds_fingerprint = ds._fingerprint
    if use_precomputed:
        cache_key = "::".join([
            f"faiss-components-v{FAISS_COMPONENT_CACHE_VERSION}",
            ds_fingerprint,
            fingerprint_column,
            str(n_neighbors),
            str(gpu),
        ])
    else:
        cache_key = "::".join([
            f"faiss-components-v{FAISS_COMPONENT_CACHE_VERSION}",
            ds_fingerprint,
            structure_column,
            input_representation,
            str(n_neighbors),
            str(gpu),
        ])
        ds = ds.map(
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

        # Infer index type from Arrow schema (reliable dtype) + first-batch
        # peek (dimension when not in schema).  Do NOT use np.asarray() on a
        # batch directly for dtype — HF Datasets returns Python int lists
        # which default to int64 regardless of the stored Arrow uint8 dtype.
        d, elem_dtype = _infer_fp_params(ds, fingerprint_column)
        # gpu_resources must stay alive for the duration of index use.
        custom_index, fp_dtype, gpu_resources = _build_faiss_index(d, elem_dtype, gpu=gpu)

        ds_fp = ds
        # GPU is transferred inside _build_faiss_index; pass device=None here
        # to avoid the HF ValueError raised when both custom_index and device
        # are supplied.
        ds_fp = ds_fp.add_faiss_index(
            column=fingerprint_column,
            index_name=f"{fingerprint_column}_idx",
            custom_index=custom_index,
            device=None,
            dtype=fp_dtype,
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
            ).astype(fp_dtype)

            # Request n_neighbors + 1 so that after excluding self we have
            # n_neighbors true neighbours.  FAISS does not guarantee that the
            # query appears first (duplicate vectors, tied distances), so we
            # filter by identity rather than skipping index 0.
            _, indices = ds_fp.search_batch(
                f"{fingerprint_column}_idx",
                batch_fp,
                k=n_neighbors + 1,
            )

            for row_of_batch, neighbor_id_list in enumerate(indices):
                row_of_dataset = start + row_of_batch
                # Filter -1 sentinels and self, then truncate to n_neighbors.
                # Truncation guards against duplicate vectors at distance zero
                # that may push self out of the top-k result, which would
                # otherwise return n_neighbors + 1 true neighbours.
                neighbors = [
                    int(j)
                    for j in neighbor_id_list
                    if j >= 0 and int(j) != row_of_dataset
                ]
                for neighbor_id in neighbors[:n_neighbors]:
                    djs.merge(row_of_dataset, neighbor_id)

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
