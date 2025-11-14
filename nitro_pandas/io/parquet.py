"""
Parquet I/O module for nitro-pandas.

This module provides functions for reading Parquet files with pandas-like
API while using Polars as the backend for high-performance I/O operations.

Key features:
- Pandas-like parameter names and behavior
- Polars backend for fast Parquet reading
- Support for lazy reading with read_parquet_lazy()
- Support for cloud storage via storage_options
"""

import polars as pl
from typing import Optional, List, Dict
from ..dataframe import DataFrame
from ..lazyframe import LazyFrame


def read_parquet(
    path: str,
    columns: Optional[List[str]] = None,
    storage_options: Optional[Dict] = None,
    n_rows: Optional[int] = None,
    parallel: str = "auto",
    dtype: Optional[dict] = None,
    **kwargs
) -> DataFrame:
    """
    Read a Parquet file into a DataFrame.
    
    This function provides a pandas-like API for reading Parquet files while
    using Polars as the backend for high-performance I/O operations.
    
    Args:
        path: Path to Parquet file (local or cloud storage)
        columns: List of column names to read (None = all columns)
        storage_options: Dictionary with storage options for cloud storage
                        (e.g., S3, GCS). Format depends on storage backend.
        n_rows: Number of rows to read (None = all rows)
        parallel: Parallelism mode ('auto' = automatic, kept for API compatibility)
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        DataFrame: DataFrame containing Parquet data
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> df = read_parquet("data.parquet", columns=["id", "name"])
        >>> df = read_parquet("s3://bucket/data.parquet", storage_options={...})
    """
    try:
        # Build Polars read_parquet parameters
        pl_kwargs = {}
        
        if columns is not None:
            pl_kwargs["columns"] = columns
        
        # Handle cloud storage options
        # Note: Polars handles storage_options differently depending on version
        # For cloud storage (S3, GCS, etc.), pass storage_options through kwargs
        if storage_options is not None:
            pl_kwargs["storage_options"] = storage_options
        
        # Note: Polars doesn't have a direct 'parallel' parameter for read_parquet
        # Parallelism is handled automatically by Polars
        # Keep parameter for API compatibility but don't pass to Polars
        if parallel != "auto":
            # Polars manages parallelism automatically
            pass
        
        # Add any other supported kwargs
        pl_kwargs.update(kwargs)
        
        # Read Parquet using Polars
        df = pl.read_parquet(path, **pl_kwargs)
        
        # Post-process n_rows if provided (limit rows after reading)
        if n_rows is not None:
            df = df.head(n_rows)
        
        # Apply dtype casting if provided
        if dtype is not None:
            for col, target_type in dtype.items():
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(target_type))
                    
    except Exception as e:
        raise ValueError(f"Error reading parquet file: {str(e)}")
    
    return DataFrame(df)


def read_parquet_lazy(
    path: str,
    columns: Optional[List[str]] = None,
    storage_options: Optional[Dict] = None,
    n_rows: Optional[int] = None,
    parallel: str = "auto",
    dtype: Optional[dict] = None,
    **kwargs
) -> LazyFrame:
    """
    Read a Parquet file lazily into a LazyFrame.
    
    Lazy reading allows Polars to optimize the query plan before execution.
    The actual reading happens when collect() is called on the LazyFrame.
    
    This is useful for large files where you want to filter/transform
    before loading all data into memory.
    
    Args:
        path: Path to Parquet file (local or cloud storage)
        columns: List of column names to read (None = all columns)
        storage_options: Dictionary with storage options for cloud storage
        n_rows: Number of rows to read (None = all rows)
        parallel: Parallelism mode ('auto' = automatic, kept for API compatibility)
        dtype: Dictionary mapping column names to data types
        **kwargs: Additional Polars-specific parameters
        
    Returns:
        LazyFrame: LazyFrame for lazy Parquet reading
        
    Raises:
        ValueError: If file cannot be read
        
    Example:
        >>> lf = read_parquet_lazy("large_data.parquet")
        >>> df = lf.filter(lf["id"] > 100).collect()
    """
    try:
        # Build Polars scan_parquet parameters
        pl_kwargs = {}
        
        # Note: scan_parquet doesn't support 'columns' directly
        # We'll handle columns selection after scanning using select()
        
        # Handle cloud storage options
        if storage_options is not None:
            pl_kwargs["storage_options"] = storage_options
        
        # Note: Polars doesn't have a direct 'parallel' parameter for scan_parquet
        # Parallelism is handled automatically by Polars
        if parallel != "auto":
            # Polars manages parallelism automatically
            pass
        
        # Add any other supported kwargs
        pl_kwargs.update(kwargs)
        
        # Scan Parquet lazily using Polars
        lf = pl.scan_parquet(path, **pl_kwargs)
        
        # Apply columns selection if provided
        # (scan_parquet doesn't support columns parameter directly)
        if columns is not None:
            lf = lf.select(columns)
        
        # Post-process n_rows and dtype if provided
        # Note: scan_parquet doesn't support these directly, so we need to
        # collect, process, then convert back to lazy
        if n_rows is not None or dtype is not None:
            df = lf.collect()
            
            # Apply n_rows limit if provided
            if n_rows is not None:
                df = df.head(n_rows)
            
            # Apply dtype casting if provided
            if dtype is not None:
                for col, target_type in dtype.items():
                    if col in df.columns:
                        df = df.with_columns(pl.col(col).cast(target_type))
            
            # Convert back to lazy
            lf = df.lazy()
            
    except Exception as e:
        raise ValueError(f"Error reading Parquet file lazily: {str(e)}")
    
    return LazyFrame(lf)
