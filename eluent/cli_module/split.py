from typing import Mapping, Optional, Union

from argparse import Namespace
import os

from carabiner import pprint_dict, print_err
from carabiner.cliutils import clicommand

from .io import _resolve_and_slice_data, _save_dataset


@clicommand("Splitting data with the following parameters")
def _split(args: Namespace) -> None:
    from datasets import concatenate_datasets, DatasetDict
    from ..utils.splitting import split_dataset
    from ..utils.splitting.utils import dataset_len
    output = args.output
    out_dir = os.path.dirname(output)
    base = os.path.basename(output)
    root, ext = os.path.splitext(base)
    if len(out_dir) > 0:
        os.makedirs(out_dir, exist_ok=True)
    
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
    if isinstance(ds, DatasetDict):
        ds = (ds,)
    print_err(f"[INFO] Generated {len(ds)} folds")
    ds_together = []
    for i, fold in enumerate(ds):
        row_counts, total_rows = {}, 0
        _ds_together = []
        for key, split_ds in fold.items():
            split_nrows = dataset_len(split_ds)
            row_counts[key] = split_nrows
            total_rows += split_nrows
            this_filename = os.path.join(out_dir, f"fold_{i}", f"{root}_{key}{ext}")
            os.makedirs(os.path.dirname(this_filename), exist_ok=True)
            _save_dataset(
                split_ds, 
                this_filename,
            )
            _ds_together.append(split_ds)
        split_fractions = {
            key: val / total_rows 
            for key, val in row_counts.items()
        }
        pprint_dict(split_fractions, message=f"Split fractions, fold {i + 1}")
        ds_together.append(concatenate_datasets(_ds_together))

    if args.plot is not None:
        from carabiner.mpl import figsaver
        from ..utils.splitting.plot import plot_chemical_splits

        for i, _ds in enumerate(ds_together):
            print_err(f"Plotting splits from {_ds}, fold {i}")
            
            (fig, axes), df = plot_chemical_splits(
                ds=_ds,
                structure_column=args.structure,
                input_representation=args.input_representation,
                split_columns="split",
                sample_size=args.plot_sample,
                additional_columns=args.extras,
                seed=args.plot_seed,
                cache=args.cache,
            )
            plot_outdir = os.path.dirname(args.plot)
            plot_root, ext = os.path.splitext(os.path.basename(args.plot))
            figsaver(
                format=ext.lstrip("."), 
                output_dir=os.path.dirname(args.plot),
            )(fig, f"{plot_root}-fold_{i}", df=df)

    return None
