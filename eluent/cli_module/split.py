from typing import Mapping, Optional, Union

from argparse import Namespace
import os

from carabiner import pprint_dict, print_err
from carabiner.cliutils import clicommand

from .io import _resolve_and_slice_data, _save_dataset


@clicommand("Splitting data with the following parameters")
def _split(args: Namespace) -> None:
    from datasets import concatenate_datasets
    from ..utils.splitting import split_dataset
    from ..utils.splitting.utils import dataset_len
    output = args.output
    out_dir = os.path.dirname(output)
    if len(out_dir) > 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if args.train is None:
        raise ValueError(f"You need to at least provide a --train fraction.")
    
    ds = _resolve_and_slice_data(
        args.input_file,
        start=args.start,
        end=args.end,
    )
    if args.type == "faiss":
        faiss_opts = {
            "cache": args.cache,
            "n_neighbors": args.n_neighbors,
        }
    else:
        faiss_opts = {}
    ds = split_dataset(
        ds=ds,
        method=args.type,
        structure_column=args.structure,
        input_representation=args.input_representation,
        train=args.train,
        validation=args.validation,
        test=args.test,
        kfolds=args.kfolds,
        batch_size=args.batch,
        seed=args.seed or 42,
        deterministic=args.seed is not None,
        **faiss_opts,
    )
    root, ext = os.path.splitext(output)
    ds_together = []
    row_counts, total_rows = {}, 0
    for key, split_ds in ds.items():
        split_nrows = dataset_len(split_ds)
        row_counts[key] = split_nrows
        total_rows += split_nrows
        _save_dataset(
            split_ds, 
            f"{root}_{key}{ext}",
        )
        ds_together.append(split_ds)
    split_fractions = {
        key: val / total_rows 
        for key, val in row_counts.items()
    }
    pprint_dict(split_fractions, message="Split fractions")
    ds_together = concatenate_datasets(ds_together)

    if args.plot is not None:

        from carabiner.mpl import figsaver
        from ..utils.splitting.plot import plot_chemical_splits

        print_err(f"Plotting splits from {ds_together}")
        
        (fig, axes), df = plot_chemical_splits(
            ds=ds_together,
            structure_column=args.structure,
            input_representation=args.input_representation,
            split_columns="split",
            sample_size=args.plot_sample,
            additional_columns=args.extras,
            seed=args.plot_seed,
            cache=args.cache,
        )
        root, ext = os.path.splitext(args.plot)
        figsaver(format=ext.lstrip("."))(fig, root, df=df)

    return None
