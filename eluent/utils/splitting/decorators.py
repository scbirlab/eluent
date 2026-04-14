"""Wrapping core algorithms with convenience."""

from typing import Callable, Mapping, Optional

from functools import wraps

def process_splits(f: Callable):

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

    """

    @wraps(f)
    def _process_splits(
        *args,
        train: float = 1.,
        validation: Optional[float] = None,
        test: Optional[float] = None,
        splits: Optional[Mapping[str, float]] = None,
        **kwargs
    ) -> Callable:
        validation = validation or (1. - train)
        test = test or (1. - train - validation)
        kwarg_splits = {
            "train": train,
        }
        if validation > 0.:
            kwarg_splits["validation"] = validation
        if test > 0.:
            kwarg_splits["test"] = test
        splits = splits or {}
        kwarg_splits.update(splits)  # override defaults, add arbitrary extra

        return f(*args, **kwargs, splits=kwarg_splits)

    return _process_splits
