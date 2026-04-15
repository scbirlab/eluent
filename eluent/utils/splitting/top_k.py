"""Approximate percentiles out-of-core."""

from typing import Dict, Iterable, Mapping, Union 

from functools import partial

from carabiner import print_err
from datasets import Dataset, IterableDataset, concatenate_datasets
import numpy as np
from tdigest import TDigest
from tqdm.auto import tqdm

from .utils import dataset_len
from ..datasets import to_dataset

def _tag_top(
    x: Mapping[str, Iterable],
    column: str,
    cutoff: float,
    output_column: str,
    delta: float = .01,
    reverse: bool = False
) -> Dict[str, Iterable]:
    out = []
    sign = -1 if reverse else 1
    for v in x[column]:
        if v >= cutoff + delta:
            out.append(sign)
        elif v <= cutoff - delta:
            out.append(-1 * sign)
        else:
            out.append(0)
    x[output_column] = out
    return x
    

def get_percentile(
    ds: Union[Dataset, IterableDataset],
    column: str,
    digest: TDigest,
    p: float,
    count: int,
    reverse: bool = False,
    delta: float = 1.,
    batch_size: int = 1024,
    cache: str = "./cache"
):
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
        _tag_top,
        fn_kwargs={
            "column": column,
            "cutoff": cutoff,
            "delta": delta,
            "output_column": top_col,
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
    maybes = to_dataset_p(ds_tfm[0], nrows=dataset_len(ds_tfm[0])).sort(column, reverse=reverse)

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
        to_dataset_p(ds_tfm[1], nrows=dataset_len(ds_tfm[1])), 
        maybes_resolved,
        to_dataset_p(ds_tfm[-1], nrows=dataset_len(ds_tfm[-1])),
    ]).map(
        lambda x: {top_col: [v == 1 for v in x[top_col]]}, 
        desc="Polishing percentile column",
        batched=True,
        batch_size=batch_size,
    )

    return final_ds


def percentiles(
    ds: Union[Dataset, IterableDataset],
    q: Mapping[str, Iterable[float]],
    compression: int = 500,
    delta: float = .01,
    reverse: bool = False,
    batch_size: int = 1024,
    cache: str = "./cache"
):
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