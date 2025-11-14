"""
Excel I/O module for nitro-pandas.

This module provides functions for reading Excel files with pandas-like
API while using Polars as the backend for high-performance I/O operations.

Key features:
- Pandas-like parameter names and behavior
- Polars backend for fast Excel reading (via fastexcel)
- Support for sheet selection by name or index
- Support for lazy reading with read_excel_lazy()
- Post-processing of pandas-like parameters (usecols, skiprows, etc.)
"""

import polars as pl
from typing import Optional, Union, List
from ..dataframe import DataFrame
from ..lazyframe import LazyFrame


def read_excel(
    path: str,
    sheet_name: Union[str, int] = 0,
    usecols: Optional[Union[List[str], str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    header: Union[int, None] = 0,
    names: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    **kwargs
) -> DataFrame:
    """
    Read an Excel file into a DataFrame.
    
    This function provides a pandas-like API for reading Excel files while
    using Polars (via fastexcel) as the backend for high-performance I/O.
    
    Parameters are automatically mapped from pandas-style names to Polars
    equivalents. Some parameters (usecols, skiprows, nrows, names) are
    post-processed after reading to match pandas behavior.
    
    Args:
        path: Path to Excel file (.xlsx, .xls)
        sheet_name: Sheet name or index to read (default: 0)
        usecols: Columns to read (list of names or indices)
        nrows: Number of rows to read (None = all rows)
        skiprows: Rows to skip (int or list of row indices)
        header: Row number to use as column names (0 = first row, None = no header)
        names: List of column names to use (overrides header)
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        DataFrame: DataFrame containing Excel data
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> df = read_excel("data.xlsx", sheet_name=0)
        >>> df = read_excel("data.xlsx", usecols=["A", "B"], nrows=1000)
    """
    try:
        # Build Polars read_excel parameters
        read_kwargs = {}
        
        # Map sheet_name to Polars format (sheet_id for int, sheet_name for str)
        if isinstance(sheet_name, int):
            read_kwargs["sheet_id"] = sheet_name
        else:
            read_kwargs["sheet_name"] = sheet_name
        
        # Add any other supported kwargs
        read_kwargs.update(kwargs)
        
        # Read Excel using Polars (requires fastexcel package)
        df = pl.read_excel(path, **read_kwargs)
        
        # Normalize possible return types (dict/list) to a single DataFrame
        # Polars may return dict of DataFrames for multiple sheets or list
        if isinstance(df, dict):
            df = next(iter(df.values()))
        elif isinstance(df, list):
            df = df[0] if len(df) > 0 else pl.DataFrame()
        
        # Post-process pandas-like parameters
        # These are applied after reading to match pandas behavior
        
        # Select columns if usecols is provided
        if isinstance(usecols, list):
            df = df.select(usecols)
        
        # Skip rows if skiprows is provided
        if isinstance(skiprows, int) and skiprows > 0:
            df = df.slice(skiprows)
        elif isinstance(skiprows, list) and skiprows:
            # Filter out specific row indices
            df = df.with_row_count("__row__").filter(
                ~pl.col("__row__").is_in(skiprows)
            ).drop("__row__")
        
        # Limit rows if nrows is provided
        if isinstance(nrows, int):
            df = df.head(nrows)
        
        # Rename columns if names is provided
        if names is not None:
            df = df.rename(dict(zip(df.columns, names)))
        
        # Apply dtype casting if provided
        if dtype is not None:
            for col, target_type in dtype.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(target_type))
                    
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")
    
    return DataFrame(df)


def read_excel_lazy(
    path: str,
    sheet_name: Union[str, int] = 0,
    usecols: Optional[Union[List[str], str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    header: Union[int, None] = 0,
    names: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    **kwargs
) -> LazyFrame:
    """
    Read an Excel file lazily into a LazyFrame.
    
    Note: This function uses eager reading then exposes as LazyFrame because
    Polars doesn't have native lazy Excel reading. The actual reading happens
    immediately, but the result is wrapped in a LazyFrame for consistency.
    
    Args:
        path: Path to Excel file (.xlsx, .xls)
        sheet_name: Sheet name or index to read (default: 0)
        usecols: Columns to read (list of names or indices)
        nrows: Number of rows to read (None = all rows)
        skiprows: Rows to skip (int or list of row indices)
        header: Row number to use as column names (0 = first row, None = no header)
        names: List of column names to use (overrides header)
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        LazyFrame: LazyFrame for lazy Excel reading
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> lf = read_excel_lazy("data.xlsx")
        >>> df = lf.filter(lf["id"] > 100).collect()
    """
    try:
        # Build Polars read_excel parameters
        read_kwargs = {}
        
        # Map sheet_name to Polars format
        if isinstance(sheet_name, int):
            read_kwargs["sheet_id"] = sheet_name
        else:
            read_kwargs["sheet_name"] = sheet_name
        
        # Add any other supported kwargs
        read_kwargs.update(kwargs)
        
        # Read Excel using Polars (eager read)
        df = pl.read_excel(path, **read_kwargs)
        
        # Normalize possible return types
        if isinstance(df, dict):
            df = next(iter(df.values()))
        elif isinstance(df, list):
            df = df[0] if len(df) > 0 else pl.DataFrame()
        
        # Post-process pandas-like parameters (same as read_excel)
        if isinstance(usecols, list):
            df = df.select(usecols)
        
        if isinstance(skiprows, int) and skiprows > 0:
            df = df.slice(skiprows)
        elif isinstance(skiprows, list) and skiprows:
            df = df.with_row_count("__row__").filter(
                ~pl.col("__row__").is_in(skiprows)
            ).drop("__row__")
        
        if isinstance(nrows, int):
            df = df.head(nrows)
        
        if names is not None:
            df = df.rename(dict(zip(df.columns, names)))
        
        # Apply dtype casting if provided
        if dtype is not None:
            for col, target_type in dtype.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(target_type))
        
        # Convert to lazy after eager reading
        lf = df.lazy()
        
    except Exception as e:
        raise ValueError(f"Error reading Excel file lazily: {str(e)}")
    
    return LazyFrame(lf)
