from argparse import Namespace
import os

from carabiner import pprint_dict, print_err
from carabiner.cliutils import clicommand

from .io import _save_dataset


@clicommand("Splitting data with the following parameters")
def _split(args: Namespace) -> None:

    from providence import DataResolver
    from providence.package_data import configure_hf_cache

    cache, _, _ = configure_hf_cache(args.cache)

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
    
    resolved = (
        DataResolver(args.input_file, cache_dir=cache)
        .select_rows(args.start, args.end)
    )

    group_columns = args.group_columns
    if group_columns is not None:
        group_column = group_columns[0] if len(group_columns) == 1 else group_columns
    else:
        group_column = "group"

    faiss_opts = {}
    if not group_columns and args.type == "faiss":
        faiss_opts = {
            "cache": cache,
            "n_neighbors": args.n_neighbors,
        }

    ds = split_dataset(
        ds=resolved.data,
        method=None if group_columns else args.type,
        group_column=group_column,
        structure_column=args.structure,
        input_representation=args.input_representation,
        train=args.train,
        validation=args.validation,
        test=args.test,
        kfolds=args.kfolds,
        batch_size=args.batch,
        seed=42 if args.seed is None else args.seed,
        deterministic=args.seed is None,
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
            if args.plot is not None and split_nrows > 0:
                if group_columns and len(group_columns) > 1:
                    if "split" in split_ds.column_names:
                        split_ds = split_ds.remove_columns("split")
                    split_ds = split_ds.add_column("split", [key] * split_nrows)
                _ds_together.append(split_ds)
        split_fractions = {
            key: count / total_rows if total_rows else 0.0
            for key, count in row_counts.items()
        }
        pprint_dict(split_fractions, message=f"Split fractions, fold {i + 1}")
        if args.plot is not None:
            ds_together.append(
                concatenate_datasets(_ds_together) 
                if len(_ds_together) > 0
                else None
            )

    if args.plot is not None:
        from carabiner.mpl import figsaver
        from ..utils.splitting.plot import plot_chemical_splits

        for i, _ds in enumerate(ds_together):
            if _ds is None:
                continue
            print_err(f"Plotting splits from {_ds}, fold {i}")
            
            (fig, axes), df = plot_chemical_splits(
                ds=_ds,
                structure_column=args.structure,
                input_representation=args.input_representation,
                split_columns="split",
                sample_size=args.plot_sample,
                additional_columns=args.extras,
                seed=args.plot_seed,
                cache=cache,
            )
            plot_outdir = os.path.dirname(args.plot)
            if len(plot_outdir) > 0:
                os.makedirs(plot_outdir, exist_ok=True)
            plot_root, ext = os.path.splitext(os.path.basename(args.plot))
            figsaver(
                format=ext.lstrip("."), 
            )(fig, os.path.join(plot_outdir, f"{plot_root}-fold_{i}"), df=df)

    return None
