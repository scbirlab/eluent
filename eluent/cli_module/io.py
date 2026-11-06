from carabiner import print_err

from ..utils.package_data import CACHE_DIR
from ..utils.splitting.utils import dataset_len


def _resolve_and_slice_data(
    data: str,
    start: int | None = None,
    end: int | None = None,
    batch_size: int = 1024,
    cache: str | None = None
):
    from datasets import DatasetDict, IterableDatasetDict
    from ..utils.datasets_io import AutoDataset

    ds = AutoDataset.load(data, cache=cache)._dataset
    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        if len(ds) != 1:
            raise ValueError(
                "Expected one dataset split. "
                f"Select a split explicitly from {list(ds)}."
            )
        ds = next(iter(ds.values()))
    nrows = dataset_len(ds)
    skip = 0 if start is None else start
    take = (nrows if end is None else end) - skip
    if (take - skip) < nrows:
        print_err(f"[INFO] Reading dataset from row {skip} to row {take + skip} / {nrows}.")
    return ds.skip(skip).take(take)


def _save_dataset(
    dataset,
    output: str
) -> None:
    print_err("[INFO] Saving dataset:\n" + str(dataset) + "\n" + f"at {output} as", end=" ")
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
