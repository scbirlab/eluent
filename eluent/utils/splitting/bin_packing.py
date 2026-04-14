"""Algorithm for packing groups into splits."""

from typing import Dict, Mapping, Optional, Union

from collections import Counter

from carabiner import print_err
from datasets import Dataset, IterableDataset
import numpy as np
from tqdm.auto import tqdm

from .utils import dataset_len

def deterministic_packing_step(
    group_size: int,
    group_name: str,
    remaining: Mapping[str, int],
    group_to_split: Mapping[str, str],
    **kwargs
):
    split_destination = max(remaining, key=remaining.__getitem__)
    group_to_split[group_name] = split_destination
    remaining[split_destination] -= group_size
    return group_to_split, remaining


def random_packing_step(
    group_size: int,
    group_name: str,
    remaining: Mapping[str, int],
    group_to_split: Mapping[str, str],
    rng
):
    eligible = [
        split_name for split_name, capacity in remaining.items() 
        if capacity >= group_size
    ]
    if len(eligible) == 0:
        # if none can fully fit, fall back to all splits (you may accept overflow)
        eligible = list(remaining)

    weights = np.array(
        [remaining[name] for name in eligible], 
        dtype=float,
    )
    probs = weights / weights.sum()

    split_destination = rng.choice(eligible, p=probs)
    group_to_split[group_name] = split_destination
    remaining[split_destination] -= group_size
    return group_to_split, remaining

def pack_bins(
    ds: Union[Dataset, IterableDataset],
    group_column: str,
    splits: Mapping[str, float],
    batch_size: int = 1024,
    num_rows: Optional[int] = None,
    deterministic: bool = True,
    seed: int = 42
) -> Dict[str, str]:
    rng = np.random.default_rng(seed=seed)
    if num_rows is None:
        num_rows = dataset_len(ds)

    sizes = Counter()
    for example in tqdm(ds.iter(batch_size=batch_size), desc="Finding unique groups"):
        sizes.update(example[group_column])

    print_err(f"There are {len(sizes)} unique groups.")
    
    group_to_split = {}
    split_target_sizes = {
        key: np.ceil(num_rows * val) for key, val in splits.items()
    }
    if deterministic:
        packing_f = deterministic_packing_step
    else:
        packing_f = random_packing_step
    for group in tqdm(sorted(
        sizes, 
        key=sizes.__getitem__, 
        reverse=True,
    ), desc="Sorting groups by size"):
        group_to_split, split_target_sizes = packing_f(
            group_size=sizes[group],
            group_name=group,
            remaining=split_target_sizes,
            group_to_split=group_to_split,
            rng=rng,
        )
            
    return group_to_split
