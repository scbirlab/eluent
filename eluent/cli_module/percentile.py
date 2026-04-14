from typing import Mapping, Optional, Union

from argparse import Namespace
import os

from carabiner import print_err
from carabiner.cliutils import clicommand

from .io import _resolve_and_slice_data, _save_dataset


@clicommand("Tagging data percentiles with the following parameters")
def _percentile(args: Namespace) -> None:

    if args.plot is not None and args.structure is None:
        raise ValueError(
            f"""
            If you want to save a plot at "{args.plot}", then you need to provide
            a chemical structure (like SMILES) column name using --structure so 
            that a UMAP embedding can be calculated.
            """
        )

    from ..utils.splitting.top_k import percentiles
    
    output = args.output
    out_dir = os.path.dirname(output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    ds = _resolve_and_slice_data(
        args.input_file,
        start=args.start,
        end=args.end,
    )
    # print(ds)
    # for b in ds:
    #     print(b)
    #     break
    q = {col: args.percentiles for col in args.columns}
    ds = percentiles(
        ds=ds,
        q=q,
        compression=args.compression,
        delta=args.delta,
        reverse=args.reverse,
        cache=args.cache,
    )
    _save_dataset(ds, output)

    if args.plot is not None:

        from carabiner.mpl import figsaver
        from ..utils.splitting.plot import plot_chemical_splits

        print_err(f"Plotting top percentiles...")

        (fig, axes), df = plot_chemical_splits(
            ds=ds,
            structure_column=args.structure,
            input_representation=args.input_representation,
            split_columns=[col for col in ds.column_names if any(f"{_q}_top_" in col for _q in q)],
            sample_size=args.plot_sample,
            additional_columns=args.extras,
            seed=args.plot_seed,
            cache=args.cache,
        )
        root, ext = os.path.splitext(args.plot)
        figsaver(format=ext.lstrip("."))(fig, root, df=df)
    return None