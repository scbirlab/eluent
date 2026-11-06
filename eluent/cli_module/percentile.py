from argparse import Namespace
import os

from carabiner import print_err
from carabiner.cliutils import clicommand

from .io import _save_dataset


@clicommand("Tagging data percentiles with the following parameters")
def _percentile(args: Namespace) -> None:

    from providence import DataResolver
    from providence.package_data import configure_hf_cache

    cache, _, _ = configure_hf_cache(args.cache)

    # if args.plot is not None and args.structure is None:
    #     raise ValueError(
    #         f"""
    #         If you want to save a plot at "{args.plot}", then you need to provide
    #         a chemical structure (like SMILES) column name using --structure so 
    #         that a UMAP embedding can be calculated.
    #         """
    #     )

    from ..utils.splitting.top_k import percentiles
    
    output = args.output
    out_dir = os.path.dirname(output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    resolved = (
        DataResolver(args.input_file, cache_dir=cache)
        .select_rows(args.start, args.end)
    )
    
    q = {col: args.percentiles for col in args.columns}
    ds = percentiles(
        ds=resolved.data,
        q=q,
        compression=args.compression,
        delta=args.delta,
        reverse=args.reverse,
        cache_dir=cache,
    )
    _save_dataset(ds, output)
    return None