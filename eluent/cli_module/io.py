from carabiner import print_err

from ..utils.package_data import CACHE_DIR


def _resolve_and_slice_data(
    data: str,
    start: int | None = None,
    end: int | None = None,
    batch_size: int = 1024,
    cache: str | None = None
):
    from datasets import DatasetDict, IterableDatasetDict
    from providence import DataResolver

    resolved = DataResolver(data, cache_dir=cache).select_rows(start, end)
    return resolved


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
