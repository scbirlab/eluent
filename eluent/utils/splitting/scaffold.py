"""Generating splits on chemical scaffold."""

from typing import Iterable, Optional, Mapping, Union

from datasets import Dataset, IterableDataset
from schemist.converting import convert_string_representation

from .bin_packing import pack_bins
from .decorators import process_splits
from .utils import annotate_split


def annotate_scaffold(
    x: Mapping[str, Iterable], 
    structure_column: str, 
    input_representation: str = 'smiles',
    scaffold_classifier: str = "scaffold",
    scaffold_column: str = "scaffold"
):
    """Annotate a chemical structure with its scaffold.

    """

    converted = convert_string_representation(
        strings=x[structure_column],
        input_representation=input_representation,
        output_representation=scaffold_classifier,
    )
    if len(x[structure_column]) > 1 and not isinstance(x[structure_column], str):
        x[scaffold_column] = list(converted)
    else:
        x[scaffold_column] = [converted]
    return x


@process_splits
def scaffold_split(
    ds: Union[Dataset, IterableDataset],
    structure_column: str,
    input_representation: str = 'smiles',
    scaffold_classifier: str = "scaffold",
    scaffold_column: str = "scaffold",
    deterministic: bool = True,
    seed: int = 42,
    batch_size: int = 1024,
    splits: Optional[Mapping[str, float]] = None
) -> Union[Dataset, IterableDataset]:

    """Spilt a dataset based on chemical scaffold (or other grouping calculation).

    """

    ds = (
        ds.
        map(
            annotate_scaffold,
            fn_kwargs={
                "structure_column": structure_column,
                "input_representation": input_representation,
                "scaffold_classifier": scaffold_classifier,
                "scaffold_column": scaffold_column,
            },
            batched=True,
            batch_size=batch_size,
            desc="Annotating scaffold",
        )
    )

    scaffold_to_split = pack_bins(
        ds,
        group_column=scaffold_column,
        splits=splits,
        batch_size=batch_size,
        deterministic=deterministic,
        seed=seed,
    )

    if isinstance(ds, Dataset):
        desc = {"desc": "Annotating splits"}
    else:
        desc = {}
    ds = ds.map(
        annotate_split,
        fn_kwargs={
            "key_to_split": scaffold_to_split,
            "key_column": scaffold_column,
            "split_column": "split",
        },
        batched=True,
        batch_size=batch_size,
        **desc,
    )

    if isinstance(ds, Dataset):
        desc = {"desc": "Generating split datasets"}
    else:
        desc = {}
    return ds, {
        key: ds.filter(lambda x: x["split"] == key, **desc)
        for key in splits
    }
