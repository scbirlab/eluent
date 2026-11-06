"""Utilities used in splitting."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
else:
    Dataset, IterableDataset = Any, Any

from carabiner import print_err
from tqdm.auto import tqdm


def save_dataset(
    dataset,
    output: str
) -> None:
    output = str(output)
    print_err(
        "[INFO] Saving dataset:\n" 
        + str(dataset) 
        + "\n" 
        + f"at {output} as", 
        end=" ",
    )
    if output.removesuffix(".gz").endswith((".csv", ".tsv", ".txt")):
        print_err("CSV.")
        dataset.to_csv(
            output, 
            sep="," if output.removesuffix(".gz").endswith(".csv") else "\t",
            compression='gzip' if output.endswith(".gz") else None,
        )
    elif output.endswith(".json"):
        print_err("JSON.")
        dataset.to_json(output)
    elif output.endswith(".parquet"):
        print_err("Parquet.")
        dataset.to_parquet(output)
    elif output.endswith(".sql"):
        print_err("SQL.")
        dataset.to_sql(output)
    elif output.endswith(".hf"):
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output)
    else:
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output + ".hf")
        print_err(f"WARNING: Unsure what format to save as for filename {output}. Defaulted to Hugging Face dataset.")
    return None


def dataset_len(ds: Dataset | IterableDataset) -> int:
    from datasets import Dataset, IterableDataset
    if isinstance(ds, Dataset):
        return ds.num_rows
    elif isinstance(ds, IterableDataset):
        print_err("Looping through dataset to count rows: ", end="")
        for i, _ in enumerate(tqdm(ds)):
            pass
        return i + 1
    else:
        raise ValueError(f"Input must be Dataset or IterableDataset but was `{type(ds)}`.")
