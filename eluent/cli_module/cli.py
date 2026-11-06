"""Command-line interface for eluent."""

from argparse import FileType
import os
import sys

from carabiner.cliutils import CLIOption, CLICommand, CLIApp

from .. import app_name, __version__

from .percentile import _percentile
from .split import _split, _write, _plot


def main() -> None:

    input_file = CLIOption(
        'input_file',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?',
        help='Input file. Default: STDIN',
    )
    input_filename = CLIOption(
        'input_file',
        type=str,
        help='Input file.',
    )
    structure_col = CLIOption(
        '--structure', '-S',
        type=str,
        default=None,
        help="""
        Column names from data file that contains a string representation 
        of chemical structure. Required for chemical model classes.
        """,
    )
    structure_representation = CLIOption(
        '--input-representation', '-R',
        type=str,
        default="smiles",
        choices=["smiles", "selfies", "inchi", "aa_seq"],
        help='Type of chemical structure string. Default: SMILES if training; for prediction, use same as training data.',
    )
    cache = CLIOption(
        '--cache',
        type=str,
        default=None,
        help='Where to cache data. Default: infer from environment/platform.',
    )
    batch_size = CLIOption(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size for training.',
    )
    checkpoint = CLIOption(
        '--checkpoint', '-c', 
        type=str,
        default=None,
        help='Checkpoint directory.',
    )
    output_name = CLIOption(
        '--output', '-o', 
        type=str,
        default=None,
        help='Output filename.',
    )
    # slice dataset
    slice_start = CLIOption(
        '--start', 
        type=int,
        default=None,
        help='First row of dataset to process.',
    )
    slice_end = CLIOption(
        '--end', 
        type=int,
        default=None,
        help='Last row of dataset to process. Default: end of dataset.',
    )

    split_type = CLIOption(
        '--type', 
        type=str,
        default="scaffold",
        choices=["random", "scaffold", "faiss"],
        help='Splitting method.',
    )
    group_columns = CLIOption(
        '--group-columns',
        nargs='+',
        default=None,
        help='Existing group columns, in axis order. Overrides --type.',
    )
    random_seed = CLIOption(
        '--seed', '-i', 
        type=int,
        default=None,
        help='Random seed. Default: determininstic.',
    )
    n_neighbors = CLIOption(
        '--n-neighbors', '-k', 
        type=int,
        default=10,
        help='Number of nearest neighbors for FAISS splitting.',
    )
    train_test_val = [
        CLIOption(
            f'--{key}', 
            type=float,
            default=None,
            help='Fraction of examples for each split. Default: infer.',
        ) for key in ("train", "validation", "test")
    ]

    kfolds = CLIOption(
        '--kfolds', '-K',
        type=int,
        default=1,
        help='Number of folds to generate training and validation. Overrides validation fraction, takes only training fraction.',
    )
    force_write_data = CLIOption(
        '--force_write_data', '-f', 
        action='store_true',
        help='Whether to save data, if remote.',
    )

    columns = CLIOption(
        '--columns', '-c',
        type=str,
        nargs='*',
        help='List of columns to tag percentiles for.',
    )
    percentiles = CLIOption(
        '--percentiles', '-p', 
        type=float,
        nargs='*',
        default=[5.],
        help='List of percentiles to calculate.',
    )
    reverse = CLIOption(
        '--reverse', '-r', 
        action='store_true',
        help='Whether to reverse percentiles (i.e. high to low).',
    )
    compression = CLIOption(
        '--compression', '-z', 
        type=int,
        default=500,
        help='How many centroids for quantile approximation. Higher is more accurate, but uses more memory.',
    )
    delta = CLIOption(
        '--delta', '-d', 
        type=float,
        default=1.,
        help='Width from percentile cutoff to buffer as borderline for refinement. Higher is more accurate, but uses more memory.',
    )
    plot_sample = CLIOption(
        '--plot-sample', '-n', 
        type=int,
        default=20_000,
        help='Subsample size for UMAP embedding.',
    )
    extras = CLIOption(
        '--extras', '-x',
        type=str,
        nargs='*',
        help='Additional columns for coloring UMAP plot.',
    )

    split = CLICommand(
        "split",
        description="Make chemical train-test-val splits on out-of-core datasets.",
        options=[
            input_filename, 
            split_type,
            kfolds,
            group_columns,
            output_name,
            checkpoint,
            force_write_data,
            slice_start,
            slice_end,
            structure_col,
            structure_representation,
            random_seed,
            cache,
            batch_size,
            n_neighbors,
        ] + train_test_val,
        main=_split,
    )

    load_chk = CLICommand(
        "write",
        description="Write a split checkpoint to tables.",
        options=[
            input_filename, 
            output_name,
            cache,
        ],
        main=_write,
    )

    plot_cmd = CLICommand(
        "plot",
        description="Visualize chemical splits.",
        options=[
            input_filename, 
            output_name,
            structure_col,
            structure_representation,
            plot_sample,
            extras,
            random_seed,
            cache,
        ],
        main=_plot,
    )

    percentiles = CLICommand(
        "percentiles",
        description="Add columns indicating whether rows are in a percentile.",
        options=[
            input_filename, 
            columns,
            percentiles,
            reverse,
            compression,
            delta,
            slice_start,
            slice_end,
            cache,
            output_name.replace(required=True),
            batch_size,
            structure_col,
            structure_representation,
        ],
        main=_percentile,
    )

    app = CLIApp(
        app_name, 
        version=__version__,
        description=(
            "Chemistry-aware splitting of large datasets."
        ),
        commands=[
            split,
            load_chk,
            percentiles,
            plot_cmd,
        ],
    )

    app.run()
    return None


if __name__ == '__main__':
    main()
