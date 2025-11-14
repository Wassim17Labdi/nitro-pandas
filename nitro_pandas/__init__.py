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

from .io import *
from .dataframe import DataFrame
from .lazyframe import LazyFrame

# Export useful Polars expressions for user convenience
from polars import (
    col, lit, when, all, any, count, min, max, mean, sum, std, var,
    first, last, concat_str
)


def concat(dfs: list[DataFrame], axis: int = 0) -> DataFrame:
    """
    Concatenate multiple DataFrames (pandas-like function).
    
    Args:
        dfs: List of DataFrames to concatenate
        axis: 0 for vertical (row-wise), 1 for horizontal (column-wise)
        
    Returns:
        DataFrame: Concatenated DataFrame
        
    Example:
        >>> import nitro_pandas as npd
        >>> df1 = npd.DataFrame({'a': [1, 2]})
        >>> df2 = npd.DataFrame({'a': [3, 4]})
        >>> result = npd.concat([df1, df2])
    """
    import polars as pl
    if axis == 0:
        out = pl.concat([d._df for d in dfs], how="vertical")
    else:
        out = pl.concat([d._df for d in dfs], how="horizontal")
    return DataFrame(out)


__all__ = [
    'DataFrame',
    'LazyFrame',
    'read_csv',
    'read_csv_lazy',
    'read_parquet',
    'read_parquet_lazy',
    'read_excel',
    'read_excel_lazy',
    'read_json',
    'read_json_lazy',
    'concat',
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
    import pandas as pd
    
    if not hasattr(pd, name):
        raise AttributeError(
            f"module 'nitro_pandas' has no attribute '{name}'. "
            f"Also not found in pandas module."
        )
    
    return getattr(pd, name)
