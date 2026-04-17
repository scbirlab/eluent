"""Wrapping core algorithms with convenience."""

from typing import Callable, Mapping, Optional
from functools import wraps

from carabiner import pprint_dict, print_err

def process_splits(fn: Callable):

    """Decorator to allow splitting functions to infer full 3-way split fractions.

    Examples
    ========
    >>> @process_splits
    ... def _echo(*args, splits=None, **kwargs):
    ...     return splits
    >>> _echo(train=0.8, validation=0.1, test=0.1) == {'train': 0.8, 'validation': 0.1, 'test': 0.1}
    True
    >>> result = _echo(train=0.8)
    >>> 'train' in result and 'validation' in result
    True
    >>> _echo(splits={'train': 0.7, 'test': 0.3})['train']
    0.7
    >>> result = _echo(train=0.8, kfolds=5)
    >>> sorted(result.keys()) == ['fold:0', 'fold:1', 'fold:2', 'fold:3', 'fold:4', 'test']
    True
    >>> abs(result['fold:0'] - 0.16) < 1e-9
    True

    """

    @wraps(fn)
    def _fn(
        *args,
        train: float = 1.,
        validation: Optional[float] = None,
        test: Optional[float] = None,
        kfolds: int = 1,
        splits: Optional[Mapping[str, float]] = None,
        **kwargs
    ) -> Callable:
        if kfolds == 1:
            validation = validation or (1. - train)
            test = test or (1. - train - validation)
            _splits = {
                "train": train,
            }
            if validation > 0.:
                _splits["validation"] = validation
            if test > 0.:
                _splits["test"] = test
        else:
            print_err(f"{kfolds=}, so ignoring {validation=}")
            test = test or (1. - train)
            _splits = {
                f"train:fold={i}": train / kfolds
                for i in range(kfolds)
            }
            if test > 0.:
                _splits["test"] = test
        _splits.update(splits or {})  # override defaults, add arbitrary extra
        split_total = sum(_splits.values())
        _splits = {key: val / split_total for key, val in _splits.items()}
        pprint_dict(_splits, message="[INFO] Inferred these split proportions")

        return fn(*args, **kwargs, splits=_splits)

    return _fn
