"""
JSON I/O module for nitro-pandas.

This module provides functions for reading JSON files with pandas-like
API while using Polars as the backend for high-performance I/O operations.

Key features:
- Pandas-like parameter names and behavior
- Polars backend for fast JSON reading
- Support for both standard JSON and JSON Lines (NDJSON) formats
- Support for lazy reading with read_json_lazy()
"""

import polars as pl
from typing import Optional, Dict
from ..dataframe import DataFrame
from ..lazyframe import LazyFrame


def read_json(
    path: str,
    schema: Optional[Dict] = None,
    infer_schema_length: Optional[int] = None,
    lines: bool = False,
    n_rows: Optional[int] = None,
    dtype: Optional[dict] = None,
    **kwargs
) -> DataFrame:
    """
    Read a JSON file into a DataFrame.
    
    This function provides a pandas-like API for reading JSON files while
    using Polars as the backend for high-performance I/O operations.
    
    Supports both standard JSON (array of objects) and JSON Lines (NDJSON)
    formats. Use lines=True for JSON Lines format.
    
    Args:
        path: Path to JSON file
        schema: Dictionary mapping column names to Polars data types
        infer_schema_length: Number of rows to use for schema inference
                            (None = use all rows)
        lines: If True, read as JSON Lines (NDJSON) format
        n_rows: Number of rows to read (None = all rows)
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        DataFrame: DataFrame containing JSON data
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> df = read_json("data.json")
        >>> df = read_json("data.ndjson", lines=True)
        >>> df = read_json("data.json", dtype={"id": "int64"})
    """
    try:
        # Build Polars read_json parameters
        pl_kwargs = {}
        
        if schema is not None:
            pl_kwargs["schema"] = schema
        
        if infer_schema_length is not None:
            pl_kwargs["infer_schema_length"] = infer_schema_length
        
        # Add any other supported kwargs
        pl_kwargs.update(kwargs)
        
        # Use read_ndjson for JSON Lines format, read_json for standard JSON
        if lines:
            df = pl.read_ndjson(path, **pl_kwargs)
        else:
            df = pl.read_json(path, **pl_kwargs)
        
        # Post-process n_rows if provided (limit rows after reading)
        if n_rows is not None:
            df = df.head(n_rows)
        
        # Apply dtype casting if provided
        if dtype is not None:
            for col, target_type in dtype.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(target_type))
                    
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {str(e)}")
    
    return DataFrame(df)


def read_json_lazy(
    path: str,
    schema: Optional[Dict] = None,
    infer_schema_length: Optional[int] = None,
    lines: bool = False,
    n_rows: Optional[int] = None,
    dtype: Optional[dict] = None,
    **kwargs
) -> LazyFrame:
    """
    Read a JSON file lazily into a LazyFrame.
    
    Note: This function uses eager reading then exposes as LazyFrame to
    avoid scan panics across different JSON formats. For true lazy reading
    of large JSON Lines files, consider using scan_ndjson directly.
    
    Args:
        path: Path to JSON file
        schema: Dictionary mapping column names to Polars data types
        infer_schema_length: Number of rows to use for schema inference
        lines: If True, read as JSON Lines (NDJSON) format
        n_rows: Number of rows to read (None = all rows)
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        LazyFrame: LazyFrame for lazy JSON reading
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> lf = read_json_lazy("data.json")
        >>> df = lf.filter(lf["id"] > 100).collect()
    """
    try:
        # Build Polars read_json parameters
        pl_kwargs = {}
        
        if schema is not None:
            pl_kwargs["schema"] = schema
        
        if infer_schema_length is not None:
            pl_kwargs["infer_schema_length"] = infer_schema_length
        
        # Add any other supported kwargs
        pl_kwargs.update(kwargs)
        
        # Use read_ndjson for JSON Lines format, read_json for standard JSON
        if lines:
            df = pl.read_ndjson(path, **pl_kwargs)
        else:
            df = pl.read_json(path, **pl_kwargs)
        
        # Post-process n_rows if provided
        if n_rows is not None:
            df = df.head(n_rows)
        
        # Apply dtype casting if provided
        if dtype is not None:
            for col, target_type in dtype.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(target_type))
        
        # Convert to lazy after eager reading
        # Note: This approach avoids scan panics with various JSON formats
        return LazyFrame(df.lazy())
        
    except Exception as e:
        raise ValueError(f"Error reading JSON file lazily: {str(e)}")
