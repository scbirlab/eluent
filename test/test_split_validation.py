from itertools import product

import pytest
from datasets import Dataset, load_dataset

from eluent.utils.splitting import SplitDataset


@pytest.fixture
def data():
    return Dataset.from_dict({
        "id": list(range(64)),
        "group": [i // 4 for i in range(64)],
    })


@pytest.mark.parametrize("kwargs", [
    {"train": 0.9, "test": 0.3},
    {"train": 1.2},
    {"train": 0.7, "validation": 0.3, "test": -0.1},
    {"train": -0.2, "test": 0.5},
    {"train": float("nan")},
    {"train": 0.8, "test": 0.3, "kfolds": 3},
    {"train": 0.8, "kfolds": 0},
    {"train": 0.5, "splits": {"test": 0.6}},
    {"train": 0., "validation": 0., "test": 0.},
    {"train": 0.8, "kfolds": True},
])
def test_group_and_split_rejects_invalid_fractions(data, kwargs):
    with pytest.raises(ValueError):
        SplitDataset(data).group_and_split(method=None, **kwargs)


@pytest.mark.parametrize("splits", [
    {"train": 0.9, "test": 0.3},
    {"train": 1.2},
    {"train": 0.8, "test": -0.1},
])
def test_split_rejects_invalid_fractions(data, splits):
    with pytest.raises(ValueError, match="Split fractions"):
        SplitDataset(data).split(splits=splits)


@pytest.mark.parametrize("kwargs, expected", [
    ({"train": 1.}, {"train"}),
    ({"train": 0.7, "test": 0.3}, {"train", "test"}),
    ({"train": 0.8, "validation": 0., "test": 0.2}, {"train", "test"}),
    ({"train": 0.75}, {"train", "validation"}),
])
def test_valid_fractions_still_accepted(data, kwargs, expected):
    result = SplitDataset(data).group_and_split(method=None, **kwargs)
    assert set(result.split_dataset["fold_x"]) == expected


def test_full_train_kfolds_has_no_test(data):
    result = SplitDataset(data).group_and_split(method=None, train=1., kfolds=2)
    assert all("test" not in fold for fold in result.split_dataset.values())


def test_empty_requested_partition_raises():
    data = Dataset.from_dict({"group": [0] * 6 + [1] * 3 + [2]})
    with pytest.raises(ValueError, match=r"'test'.*0\.05.*3 groups"):
        SplitDataset(data).group_and_split(method=None, train=0.95, test=0.05)


def test_empty_fold_raises():
    data = Dataset.from_dict({"group": [0] * 5 + [1] * 5})
    with pytest.raises(ValueError, match="train:fold=2"):
        SplitDataset(data).group_and_split(method=None, train=1., kfolds=3)


def test_empty_multiaxis_intersection_raises():
    # Each axis packs 40/60, but train is A1 & B1, which has no rows.
    rows = [("A0", "B0")] * 20 + [("A0", "B1")] * 40 + [("A1", "B0")] * 40
    data = Dataset.from_dict({"a": [a for a, _ in rows], "b": [b for _, b in rows]})
    with pytest.raises(ValueError, match=r"\['fold_x/train'\] are empty"):
        SplitDataset(data).group_and_split(
            method=None, group_column=["a", "b"], train=0.4, test=0.6,
        )


def test_undersized_partition_warns():
    data = Dataset.from_dict({"group": [0] * 80 + [1] * 16 + [2] * 4})
    with pytest.warns(UserWarning, match="'test' has 4 rows"):
        SplitDataset(data).group_and_split(
            method=None, train=0.7, validation=0.2, test=0.1,
        )


@pytest.fixture
def checkpoint(tmp_path):
    rows = list(product(range(4), repeat=3))
    source = tmp_path / "source.parquet"
    Dataset.from_dict({
        "id": list(range(len(rows))),
        "scaffold": [row[0] for row in rows],
    }).to_parquet(str(source))
    path = tmp_path / "split.el"
    SplitDataset(str(source)).group_and_split(
        method=None, group_column="scaffold", train=0.5, test=0.5,
    ).save(path)
    return path


def test_checkpoint_rejects_reordered_partition(tmp_path, checkpoint):
    partition_path = checkpoint / "partition.parquet"
    partition = load_dataset("parquet", data_files=str(partition_path), split="train")
    partition.shuffle(seed=1).flatten_indices().to_parquet(str(partition_path))
    with pytest.raises(ValueError, match="has changed"):
        SplitDataset.load(checkpoint, cache_dir=str(tmp_path / "cache"))


@pytest.mark.parametrize("kwargs, clash", [
    ({"method": None, "group_column": "g", "train": 0.75, "test": 0.25}, "test"),
    ({"method": None, "group_column": "g", "train": 0.75, "kfolds": 2}, "train:fold=0_0|train"),
    ({"method": "random", "train": 0.75, "test": 0.25}, "group"),
])
def test_input_columns_are_never_overwritten(kwargs, clash):
    data = Dataset.from_dict({"g": list(range(8)), clash: list(range(8))})
    with pytest.raises(ValueError, match=rf"Input columns \['{clash}'\] clash"):
        SplitDataset(data).group_and_split(**kwargs)


def test_unrequested_partition_name_passes_through():
    data = Dataset.from_dict({"g": list(range(8)), "test": list(range(8))})
    result = SplitDataset(data).group_and_split(
        method=None, group_column="g", train=0.75, validation=0.25, test=0.,
    )
    for part in result.split_dataset["fold_x"].values():
        assert part["test"] == part["g"]
