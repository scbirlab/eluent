"""Approximate percentiles out-of-core."""

from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Union 

from functools import partial

from carabiner import print_err
if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
else:
    Dataset, IterableDataset = Any, Any
import numpy as np
from tdigest import TDigest
from tqdm.auto import tqdm

from .utils import dataset_len
from ..datasets import to_dataset
from .grouping import tag_top
    

def get_percentile(
    ds: Dataset | IterableDataset,
    column: str,
    digest: TDigest,
    p: float,
    count: int,
    reverse: bool = False,
    delta: float = 1.,
    batch_size: int = 1024,
    cache: str | None = None
):
    from datasets import Dataset, IterableDataset, concatenate_datasets
    if not reverse:
        cutoff = digest.percentile(100. - p)
    else:
        cutoff = digest.percentile(p)
    k = np.ceil(count * p / 100.).astype(int)
    print_err(f"Approx cutoff={cutoff:.4f}; need top {k} of {count} examples")

    top_col = f"{column}_top_{int(p)}_pc"
    if isinstance(ds, Dataset):
        desc = {"desc": "Tagging definitelys and maybes"}
    else:
        desc = {}
    ds = ds.map(
        tag_top,
        fn_kwargs={
            "key": column,
            "cutoff": cutoff,
            "delta": delta,
            "column": top_col,
            "reverse": reverse,
        },
        batched=True,
        batch_size=batch_size,
        **desc,
    )
    
    ds_tfm = {}
    desc = "Separating definitelys from maybes ({})"
    for v in (1, -1, 0):
        if isinstance(ds, Dataset):
            _desc = {"desc": desc.format(v)}
        else:
            _desc = {}
        ds_tfm[v] = ds.filter(lambda x: x[top_col] == v, **_desc)
    
    n_definitely_true = dataset_len(ds_tfm[1])
    n_required_from_maybe = k - n_definitely_true
    
    if isinstance(ds, Dataset):
        desc = {"desc": "Tagging borderline"}
    else:
        desc = {}
    to_dataset_p = partial(to_dataset, batch_size=batch_size, cache=cache)
    maybes = to_dataset_p(ds_tfm[0], nrows=dataset_len(ds_tfm[0])).sort(column, reverse=not reverse)

    print_err(f"Confidently tagged {n_definitely_true} {column=}, {p=}; need {n_required_from_maybe} / {maybes.num_rows} to make {k} total.")
    maybes_to_true = maybes.take(n_required_from_maybe).map(lambda x: {top_col: 1}, desc="Tagging maybes as True")
    try:
        maybes_to_false = maybes.skip(n_required_from_maybe).map(lambda x: {top_col: -1}, desc="Tagging maybes as False")
    except IndexError:
        maybes_resolved = maybes_to_true
    else:
        maybes_resolved = concatenate_datasets([
            maybes_to_true,
            maybes_to_false,
        ])
    final_ds = concatenate_datasets([
        to_dataset_p(ds_tfm[1]), 
        maybes_resolved,
        to_dataset_p(ds_tfm[-1]),
    ]).map(
        lambda x: {top_col: [v == 1 for v in x[top_col]]}, 
        desc="Polishing percentile column",
        batched=True,
        batch_size=batch_size,
    )

    return final_ds


def percentiles(
    ds: Dataset | IterableDataset,
    q: Mapping[str, Iterable[float]],
    compression: int = 500,
    delta: float = .01,
    reverse: bool = False,
    batch_size: int = 1024,
    cache: str | None = None
):
    """Tag rows in the top (or bottom) percentiles of columns, out-of-core.

    Examples
    ========
    >>> import numpy as np
    >>> from datasets import Dataset
    >>> x = np.random.default_rng(0).normal(size=3000)
    >>> ds = Dataset.from_dict({"x": x})
    >>> out = percentiles(ds, q={"x": [5]}, delta=1.)
    >>> tagged = np.array(out["x"])[np.array(out["x_top_5_pc"], dtype=bool)]
    >>> len(tagged)
    150
    >>> bool((tagged >= np.quantile(x, .95)).all())
    True
    >>> out = percentiles(ds, q={"x": [5]}, delta=1., reverse=True)
    >>> tagged = np.array(out["x"])[np.array(out["x_top_5_pc"], dtype=bool)]
    >>> len(tagged)
    150
    >>> bool((tagged <= np.quantile(x, .05)).all())
    True

    """
    
    from datasets import Dataset, IterableDataset
    digests = {
        key: TDigest(K=compression) for key in q
    }
    input_ds = ds
    for i, example in enumerate(tqdm(ds.iter(batch_size=batch_size), desc="Building quantile sketch")):
        # print(example)
        for key, digest in digests.items():
            # print(example[key])
            digest.batch_update(example[key])
    count = i * batch_size + len(example[key])

    for key, percentiles in q.items():
        for percentile in percentiles:
            ds = get_percentile(
                ds=ds,
                column=key,
                digest=digests[key],
                p=percentile,
                count=count,
                delta=delta,
                batch_size=batch_size,
                reverse=reverse,
                cache=cache,
            )
    if isinstance(input_ds, Dataset):
        return ds
    else:
        return ds.to_iterable_dataset()
    