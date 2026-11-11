"""
nitro-pandas: A high-performance pandas-like DataFrame library powered by Polars.

This package provides a pandas-compatible API while using Polars as the backend
for optimized data operations. It combines the familiar pandas syntax with
Polars' performance benefits.

Key features:
- Pandas-like API for familiar usage
- Polars backend for high-performance operations
- Support for lazy evaluation with LazyFrame
- Automatic fallback to pandas for unimplemented methods
- Comprehensive I/O support (CSV, Parquet, JSON, Excel)

Main classes:
- DataFrame: Pandas-like DataFrame wrapper around Polars
- LazyFrame: Pandas-like LazyFrame wrapper for lazy evaluation

Example:
    >>> import nitro_pandas as npd
    >>> df = npd.read_csv("data.csv")
    >>> result = df.loc[df['id'] > 2]
    >>> df.groupby('category')['value'].mean()
"""

import functools
import inspect

import pandas as pd
import polars as pl

from .io import *
from .dataframe import DataFrame, PandasFallbackWarning, Series, _from_pandas, _pandas_arg, _warn_fallback
from .lazyframe import LazyFrame
from .profiling import profile_compare

# Export useful Polars expressions for user convenience
from polars import (
    col, lit, when, all, any, count, min, max, mean, sum, std, var,
    first, last, concat_str
)


def _polars_frame(obj):
    if isinstance(obj, Series):
        return obj._series.to_frame()
    if isinstance(obj, pd.DataFrame):
        return _from_pandas(obj)
    obj._flush_pending()
    return obj._df


def concat(objs, axis=0, ignore_index=False, **kwargs):
    """Concatenate DataFrames or Series; extra pandas options go through pandas."""
    if kwargs:
        _warn_fallback("concat")
        return pd.concat(_pandas_arg(list(objs)), axis=axis, ignore_index=ignore_index, **kwargs)
    frames = [_polars_frame(o) for o in objs]
    if axis in (1, "columns"):
        return DataFrame(pl.concat(frames, how="horizontal"))
    same_schema = len({tuple(f.schema.items()) for f in frames}) == 1
    return DataFrame(pl.concat(frames, how="vertical" if same_schema else "diagonal_relaxed"))


def merge(left, right, **kwargs):
    """Module-level merge, as pd.merge."""
    frame = DataFrame(_from_pandas(left)) if isinstance(left, pd.DataFrame) else left
    return frame.merge(right, **kwargs)


__all__ = [
    'DataFrame',
    'LazyFrame',
    'PandasFallbackWarning',
    'read_csv',
    'read_csv_lazy',
    'read_parquet',
    'read_parquet_lazy',
    'read_excel',
    'read_excel_lazy',
    'read_json',
    'read_json_lazy',
    'concat',
    'merge',
    'profile_compare',
    # Polars expressions
    'col', 'lit', 'when', 'all', 'any', 'count', 'min', 'max', 'mean',
    'sum', 'std', 'var', 'first', 'last', 'concat_str',
]


def __getattr__(name: str):
    """
    Automatic fallback to pandas for unimplemented module-level functions.
    
    This enables access to pandas functions at the package level that are
    not explicitly implemented in nitro-pandas. For example:
    - npd.describe() → pd.describe()
    - npd.get_dummies() → pd.get_dummies()
    
    Args:
        name: Function or attribute name from pandas
        
    Returns:
        Function or attribute from pandas module
        
    Raises:
        AttributeError: If attribute doesn't exist in pandas either
    """
    if not hasattr(pd, name):
        raise AttributeError(
            f"module 'nitro_pandas' has no attribute '{name}'. "
            f"Also not found in pandas module."
        )

    attr = getattr(pd, name)
    if not inspect.isfunction(attr):
        return attr

    @functools.wraps(attr)
    def fallback(*args, **kwargs):
        _warn_fallback(name)
        return attr(*_pandas_arg(list(args)), **_pandas_arg(kwargs))

    return fallback
