
from itertools import product

import pytest
from datasets import Dataset

from eluent.utils.splitting import SplitDataset


@pytest.fixture
def source_file(tmp_path):
    axes = ("scaffold", "taxon", "target")
    rows = list(product(range(4), repeat=3))

    dataset = Dataset.from_dict({
        "id": list(range(len(rows))),
        **{
            axis: [row[i] for row in rows]
            for i, axis in enumerate(axes)
        },
    })

    path = tmp_path / "source.parquet"
    dataset.to_parquet(str(path))
    return path


@pytest.mark.parametrize("kwargs", [
    {
        "group_column": ("scaffold",),
        "train": 0.5,
        "test": 0.5,
    },
    {
        "group_column": ("scaffold", "taxon", "target"),
        "train": 0.75,
        "test": 0.25,
        "kfolds": 3,
    },
])
def test_checkpoint_roundtrip(tmp_path, source_file, kwargs):
    original = SplitDataset(str(source_file)).group_and_split(
        method=None,
        **kwargs,
    )

    checkpoint = tmp_path / "split.el"
    original.save(checkpoint)

    restored = SplitDataset.load(
        checkpoint,
        cache_dir=str(tmp_path / "cache"),
    )

    # Same fold and partition names
    assert set(original.split_dataset) == set(restored.split_dataset)

    for fold_key, original_fold in original.split_dataset.items():
        restored_fold = restored.split_dataset[fold_key]

        assert set(original_fold) == set(restored_fold)

        for split_key, original_part in original_fold.items():
            restored_part = restored_fold[split_key]

            # Exact membership AND ordering
            assert (
                list(original_part["id"])
                == list(restored_part["id"])
            )

    assert original.split_info.counts == restored.split_info.counts
    assert original.split_info.fractions == restored.split_info.fractions



def test_checkpoint_rejects_length_mismatch(tmp_path, source_file):
    splitter = SplitDataset(str(source_file)).group_and_split(
        method=None,
        group_column="scaffold",
        train=0.5,
        test=0.5,
    )

    checkpoint = tmp_path / "split.el"
    splitter.save(checkpoint)

    # Deliberately corrupt the partition table
    partition_path = checkpoint / "partition.parquet"

    from datasets import load_dataset

    partition = load_dataset(
        "parquet",
        data_files=str(partition_path),
        split="train",
    )

    partition = partition.select(range(len(partition) - 1))
    partition.to_parquet(str(partition_path))

    with pytest.raises(ValueError, match="partition length"):
        SplitDataset.load(
            checkpoint,
            cache_dir=str(tmp_path / "cache"),
        )
