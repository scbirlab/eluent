"""Wrapping core algorithms with convenience."""

from typing import Callable, Mapping, Optional

from functools import wraps

def process_splits(f: Callable):

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
