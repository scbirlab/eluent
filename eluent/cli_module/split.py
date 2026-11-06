from argparse import Namespace
import os
from pathlib import Path

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
        ds=resolved,
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

    if args.output is None and args.checkpoint is None:
        raise ValueError("At least one of --output and --checkpoint should be provided.")
    
    if args.output is not None and isinstance(args.output, str):
        output = args.output
        out_dir = os.path.dirname(output)
        base = os.path.basename(output)
        root, ext = os.path.splitext(base)
        ds.write_tables(
            Path(out_dir) / root,
            format=ext,
        )
        pprint_dict(
            ds.split_info.counts, 
            message="Split counts",
        )
        pprint_dict(
            ds.split_info.fractions, 
            message="Split fractions",
        )
    if args.checkpoint is not None and isinstance(args.checkpoint, str):
        ds.save(
            args.checkpoint,
            save_source_data=args.force_write_data,
        )
    return None


@clicommand("Writing checkpoint to tables with the following parameters")
def _write(args: Namespace) -> None:

    from providence.package_data import configure_hf_cache

    cache, _, _ = configure_hf_cache(args.cache)

    from ..utils.splitting import SplitDataset

    ds = SplitDataset.load(
        args.input_file,
        cache_dir=args.cache,
    )
    
    if args.output is not None and isinstance(args.output, str):
        output = args.output
        out_dir = os.path.dirname(output)
        base = os.path.basename(output)
        root, ext = os.path.splitext(base)
        ds.write_tables(
            Path(out_dir) / root,
            format=ext,
        )
        pprint_dict(
            ds.split_info.counts, 
            message="Split counts",
        )
        pprint_dict(
            ds.split_info.fractions, 
            message="Split fractions",
        )
    return None


@clicommand("Plot chemical splits with the following parameters")
def _plot(args: Namespace) -> None:

    from providence.package_data import configure_hf_cache

    cache, _, _ = configure_hf_cache(args.cache)

    from carabiner.mpl import figsaver
    from ..utils.splitting.plot import plot_chemical_splits

    plot_outdir = os.path.dirname(args.output)
    if len(plot_outdir) > 0:
        os.makedirs(plot_outdir, exist_ok=True)
    plot_root, ext = os.path.splitext(os.path.basename(args.output))
    _figsaver = figsaver(
        format=ext.lstrip("."), 
    )

    if args.mode == "split":
        from ..utils.splitting import SplitDataset
        ds = SplitDataset.load(
            args.input_file,
            cache_dir=args.cache,
        )

        if args.structure is not None:
            from datasets import concatenate_datasets
            for split_key, _ds in ds.split_dataset.items():
                if _ds is None or len(_ds) == 0:
                    continue
                n_rows = [d.num_rows for _, d in _ds.items()]
                if any(n > 1_000_000 for n in n_rows):
                    print_err(
                        f"[WARN] Skipping {split_key} plots, too many rows: {n_rows}"
                    )
                    continue
                print_err(f"Plotting splits from {split_key}: {_ds}")
                _ds = concatenate_datasets([
                    d.map(
                        lambda x, c=k: x | {"split": k},
                    ) for k, d in _ds.items()
                ])
                (fig, axes), df = plot_chemical_splits(
                    ds=_ds,
                    structure_column=args.structure,
                    input_representation=args.input_representation,
                    split_columns="split",
                    sample_size=args.plot_sample,
                    additional_columns=args.extras,
                    seed=args.seed,
                    cache=cache,
                )
                _figsaver(
                    fig, 
                    os.path.join(
                        plot_outdir, 
                        f"{plot_root}-{split_key}"), 
                        df=df,
                    )
    elif args.mode == "percentiles":
        ds = DataResolver(
            args.input_file, 
            cache_dir=cache,
        )

        (fig, axes), df = plot_chemical_splits(
            ds=ds.data,
            structure_column=args.structure,
            input_representation=args.input_representation,
            split_columns=[
                col for col in ds.column_names 
                if any(
                    f"{_q}_top_" in col 
                    for _q in q
                )
            ],
            sample_size=args.plot_sample,
            additional_columns=args.extras,
            seed=args.plot_seed,
            cache=cache,
        )
        root, ext = os.path.splitext(args.plot)
        _figsaver(fig, root, df=df)

    return None


