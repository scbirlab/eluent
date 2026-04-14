"""For plotting splits."""

from typing import Iterable, Optional, Union

from datasets import Dataset
from carabiner import cast, print_err
from carabiner.mpl import add_legend, grid
import numpy as np
from schemist.converting import convert_string_representation
from schemist.features import calculate_feature
from schemist.tables import converter
import scipy
import umap

from .utils import dataset_len
from ..datasets import to_dataset 


def _check_columns(
    dataset: Dataset,
    columns: Optional[Iterable[str]] = None
):
    if columns is None:
        columns = []
    else:
        missing_from_data = [
            col for col in columns 
            if col not in dataset.column_names
        ]
        if len(missing_from_data) > 0:
            raise KeyError(f"Columns missing from dataset: {', '.join(missing_from_data)}")
    return columns


def plot_chemical_splits(
    ds: Dataset,
    structure_column: str = 'smiles',
    input_representation: str = 'smiles',
    split_columns: Optional[Iterable[str]] = None,
    descriptors: Optional[Union[str, Iterable[str]]] = None,
    sample_size: int = 20_000,
    additional_columns: Optional[Union[str, Iterable[str]]] = None,
    seed: int = 42,
    cache: str = "./cache",
    clean_smiles_column: str = "__clean_smiles__"
):

    """

    """
    
    if descriptors is None:
        descriptors = ['mwt', 'clogp', 'tpsa', 'max_charge', 'min_charge']
    else:
        descriptors = cast(descriptors, to=list)

    if split_columns is None:
        split_columns = ["split"]
    else:
        split_columns = cast(split_columns, to=list)
    if additional_columns is None:
        additional_columns = []
    else:
        additional_columns = cast(additional_columns, to=list)

    additional_columns = _check_columns(ds, additional_columns) + [col for col in descriptors if col in ds.column_names]
    descriptors = [col for col in descriptors if col not in ds.column_names]
    split_columns = _check_columns(ds, split_columns)

    cols_to_plot = sorted(set(descriptors + additional_columns + split_columns))

    print_err(
        f"Plotting UMAP embedding with maximum {sample_size} rows, coloring by", 
        ", ".join(cols_to_plot)
    )

    if dataset_len(ds) > sample_size:
        if isinstance(ds, Dataset):
            ds = ds.to_iterable_dataset(num_shards=min(1024, dataset_len(ds)))
        ds = ds.shuffle(seed=seed, buffer_size=1024).take(sample_size)
        ds = to_dataset(ds, nrows=sample_size, cache=cache)

    ds = ds.map(
        lambda x: {
            clean_smiles_column: convert_string_representation(
                strings=x[structure_column],
                input_representation=input_representation,
                output_representation="smiles",
            )
        },
    )

    df = (
        ds
        .to_pandas(batched=False)
        .assign(points=lambda x: np.nan * np.ones((x.shape[0], )))
    )
    errors, df = converter(
        df, 
        output_representation=descriptors,
    )
    fps, _ = calculate_feature(
        strings=df[clean_smiles_column],
        feature_type="fp",
        on_bits=True,
        return_dataframe=False,
    )
    fps = [sorted(map(int, fp.split(";"))) for fp in fps]
    fp_matrix = scipy.sparse.lil_matrix(
        (len(fps), 2048), 
        dtype=np.float32,
    )
    fp_matrix.rows = np.array(fps, dtype=object)
    fp_matrix.data = np.array(
        [np.ones(len(row), dtype=np.float32).tolist() for row in fps],
        dtype=object
    )
    fp_matrix = fp_matrix.tocsr()

    umapper = umap.UMAP(
        metric='jaccard', 
        min_dist=.6,
        random_state=seed, 
        low_memory=True,
    )
    umapper.fit(fp_matrix)
    fp_umapped = umapper.transform(fp_matrix)
    df = df.assign(
        umap1=fp_umapped[:,0], 
        umap2=fp_umapped[:,1],
    )

    fig, axes = grid(
        ncol=len(cols_to_plot) + 1, 
        aspect_ratio=1.2,
        sharex='row', 
        sharey='row', 
    )
    common_plot_kwargs = {
        "x": "umap1",
        "y": "umap2",
        "s": 1., 
    }
        
    for ax, col in zip(axes, ["points"] + cols_to_plot):
        ax.scatter(
            **common_plot_kwargs,
            data=df,
            c='lightgrey',
            zorder=-5,
        )
        if col in split_columns:
            unique_labels = sorted(df[col].unique())
            binary = (len(unique_labels) == 2 and (unique_labels[0] in (True, 0)))
            for i, label in enumerate(unique_labels):
                ax.scatter(
                    **(common_plot_kwargs | ({"s": 5.} if (label in (True, 1) and binary) else {})),
                    c="lightgrey" if (label in (False, 0) and binary) else f"C{i}",
                    data=df.query(f"{col} == @label"),
                    zorder=1,
                    label="_none" if (label in (False, 0) and binary) else label,
                )
            add_legend(ax)
        else:
            non_na_vals = df[col][~df[col].isna()]
            if all([
                np.all(non_na_vals >= 0.),
                len(non_na_vals) > 0,
                col not in ('min_charge', 'max_charge', 'points'),
                non_na_vals.min() != non_na_vals.max(),
            ]):
                log_colors = True
            else:
                log_colors = False
            sc = ax.scatter(
                **common_plot_kwargs,
                c=col,
                data=df,
                cmap='magma', 
                norm="log" if log_colors else None,
                zorder=1,
            )
            if col != 'points':
                try:
                    fig.colorbar(sc, ax=ax)
                except ValueError as e:
                    print_err(e)
        ax.set(title=col)
        ax.set_axis_off()
    
    return (fig, axes), df
