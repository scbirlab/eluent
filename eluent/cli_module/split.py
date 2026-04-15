from typing import Mapping, Optional, Union

from argparse import Namespace
import os

from carabiner import print_err
from carabiner.cliutils import clicommand

from .io import _resolve_and_slice_data, _save_dataset


@clicommand("Splitting data with the following parameters")
def _split(args: Namespace) -> None:

    from ..utils.splitting import split_dataset
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
    ds, splits = split_dataset(
        ds=ds,
        method=args.type,
        structure_column=args.structure,
        input_representation=args.input_representation,
        train=args.train,
        validation=args.validation,
        test=args.test,
        batch_size=args.batch,
        seed=args.seed or 42,
        deterministic=args.seed is not None,
        **faiss_opts,
    )
    root, ext = os.path.splitext(output)
    for key, split_ds in splits.items():
        _save_dataset(
            split_ds, 
            f"{root}_{key}{ext}",
        )

    if args.plot is not None:

        from carabiner.mpl import figsaver
        from ..utils.splitting.plot import plot_chemical_splits

        print_err(f"Plotting splits...")
        
        (fig, axes), df = plot_chemical_splits(
            ds=ds,
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
