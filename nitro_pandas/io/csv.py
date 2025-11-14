"""
CSV I/O module for nitro-pandas.

This module provides functions for reading and writing CSV files with
pandas-like API while using Polars as the backend for performance.

Key features:
- Pandas-like parameter names and behavior
- Polars backend for fast CSV reading/writing
- Support for lazy reading with read_csv_lazy()
- Automatic parameter mapping from pandas to Polars
"""

import polars as pl
from typing import Optional, Union, List
from ..dataframe import DataFrame
from ..lazyframe import LazyFrame


def read_csv(
    filepath_or_buffer: str,
    sep: str = ",",
    delimiter: Optional[str] = None,
    header: Union[int, str, None] = "infer",
    names: Optional[List[str]] = None,
    usecols: Optional[Union[List[str], List[int]]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    na_values: Optional[Union[str, List[str], dict]] = None,
    encoding: Optional[str] = None,
    comment_prefix: Optional[str] = None,
    decimal_comma: bool = False,
    truncate_ragged_lines: bool = False,
    dtype: Optional[dict] = None,
    **kwargs
) -> DataFrame:
    """
    Read a CSV file into a DataFrame.
    
    This function provides a pandas-like API for reading CSV files while
    using Polars as the backend for high-performance I/O operations.
    
    Parameters are automatically mapped from pandas-style names to Polars
    equivalents. Unsupported parameters are ignored.
    
    Args:
        filepath_or_buffer: Path to CSV file or file-like object
        sep: Field separator (default: ',')
        delimiter: Alternative field separator (overrides sep)
        header: Row number(s) to use as column names (default: 'infer')
        names: List of column names to use
        usecols: Columns to read (list of names or indices)
        nrows: Number of rows to read
        skiprows: Rows to skip (int or list of row indices)
        na_values: Values to interpret as null
        encoding: File encoding (e.g., 'utf-8', 'latin-1')
        comment_prefix: Character(s) indicating comment lines
        decimal_comma: Use comma as decimal separator
        truncate_ragged_lines: Truncate lines with inconsistent column counts
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        DataFrame: DataFrame containing CSV data
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> df = read_csv("data.csv", sep=",", nrows=1000)
        >>> df = read_csv("data.csv", dtype={"id": "int64", "name": "str"})
    """
    # Map pandas-style parameters to Polars equivalents
    separator = delimiter if delimiter is not None else sep
    
    # Handle header parameter
    if header == "infer":
        has_header = True
    elif header is None:
        has_header = False
    else:
        has_header = True
        # If header is an integer > 0, skip those rows
        if isinstance(header, int) and header > 0:
            skiprows = (
                list(range(header))
                if skiprows is None
                else list(range(header)) + (skiprows if isinstance(skiprows, list) else [skiprows])
            )
    
    # Convert skiprows to skip_rows format
    skip_rows = 0
    if isinstance(skiprows, int):
        skip_rows = skiprows
    elif isinstance(skiprows, list) and skiprows:
        skip_rows = max(skiprows) + 1
    
    try:
        # Build Polars read_csv parameters
        pl_kwargs = {
            "separator": separator,
            "has_header": has_header,
            "n_rows": nrows,
            "skip_rows": skip_rows,
            "comment_prefix": comment_prefix,
            "decimal_comma": decimal_comma,
            "truncate_ragged_lines": truncate_ragged_lines,
        }
        
        # Add optional parameters only if provided
        if encoding is not None:
            pl_kwargs["encoding"] = encoding
        
        # Handle column selection
        if usecols is not None:
            pl_kwargs["columns"] = usecols
        
        # Handle null value specification
        if na_values is not None:
            if isinstance(na_values, list):
                pl_kwargs["null_values"] = na_values
            elif isinstance(na_values, str):
                pl_kwargs["null_values"] = [na_values]
            elif isinstance(na_values, dict):
                # Polars doesn't support per-column null values like pandas
                # Skip dict na_values for now
                pass
        
        # Add any additional Polars-specific parameters
        pl_kwargs.update(kwargs)
        
        # Read CSV using Polars
        df = pl.read_csv(filepath_or_buffer, **pl_kwargs)
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Apply dtype casting if provided
    if dtype is not None:
        for col, target_type in dtype.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(target_type))
    
    # Apply column names if provided
    if names is not None:
        df.columns = names
    
    return DataFrame(df)


def read_csv_lazy(
    filepath_or_buffer: str,
    sep: str = ",",
    delimiter: Optional[str] = None,
    has_header: bool = True,
    skip_rows: int = 0,
    columns: Optional[Union[List[str], List[int]]] = None,
    n_rows: Optional[int] = None,
    null_values: Optional[Union[str, List[str]]] = None,
    encoding: Optional[str] = None,
    comment_prefix: Optional[str] = None,
    decimal_comma: bool = False,
    dtype: Optional[dict] = None,
    **kwargs
) -> LazyFrame:
    """
    Read a CSV file lazily into a LazyFrame.
    
    Lazy reading allows Polars to optimize the query plan before execution.
    The actual reading happens when collect() is called on the LazyFrame.
    
    This is useful for large files where you want to filter/transform
    before loading all data into memory.
    
    Args:
        filepath_or_buffer: Path to CSV file or file-like object
        sep: Field separator (default: ',')
        delimiter: Alternative field separator (overrides sep)
        has_header: Whether file has a header row
        skip_rows: Number of rows to skip
        columns: Columns to read (list of names or indices)
        n_rows: Number of rows to read
        null_values: Values to interpret as null
        encoding: File encoding (e.g., 'utf-8', 'latin-1')
        comment_prefix: Character(s) indicating comment lines
        decimal_comma: Use comma as decimal separator
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        LazyFrame: LazyFrame for lazy CSV reading
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> lf = read_csv_lazy("large_data.csv")
        >>> df = lf.filter(lf["id"] > 100).collect()
    """
    separator = delimiter if delimiter is not None else sep
    
    try:
        # Build Polars scan_csv parameters
        pl_kwargs = {
            "separator": separator,
            "has_header": has_header,
            "skip_rows": skip_rows,
            "comment_prefix": comment_prefix,
            "decimal_comma": decimal_comma,
        }
        
        # Add optional parameters only if provided
        if encoding is not None:
            pl_kwargs["encoding"] = encoding
        
        # Add optional parameters if provided (only those supported by scan_csv)
        if n_rows is not None:
            pl_kwargs["n_rows"] = n_rows
        
        if null_values is not None:
            if isinstance(null_values, list):
                pl_kwargs["null_values"] = null_values
            elif isinstance(null_values, str):
                pl_kwargs["null_values"] = [null_values]
        
        # Add any other supported kwargs
        pl_kwargs.update(kwargs)
        
        # Scan CSV lazily using Polars
        lf = pl.scan_csv(filepath_or_buffer, **pl_kwargs)
        
        # Apply dtype casting if provided
        # Note: scan_csv doesn't support dtype directly, so we need to
        # collect, cast, then convert back to lazy
        if dtype is not None:
            df = lf.collect()
            for col, target_type in dtype.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(target_type))
            lf = df.lazy()
            
    except Exception as e:
        raise ValueError(f"Error reading CSV file lazily: {str(e)}")
    
    return LazyFrame(lf)
