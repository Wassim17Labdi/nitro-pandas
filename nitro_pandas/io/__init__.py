"""
I/O module for nitro-pandas.

This module provides functions for reading and writing data files in various
formats (CSV, Parquet, JSON, Excel) with pandas-like API while using Polars
as the backend for high-performance I/O operations.

All read functions return nitro-pandas DataFrame or LazyFrame objects.
All write operations are performed via DataFrame.to_* methods.
"""

from .csv import read_csv, read_csv_lazy
from .parquet import read_parquet, read_parquet_lazy
from .excel import read_excel, read_excel_lazy
from .json import read_json, read_json_lazy

__all__ = [
    'read_csv',
    'read_csv_lazy',
    'read_parquet',
    'read_parquet_lazy',
    'read_excel',
    'read_excel_lazy',
    'read_json',
    'read_json_lazy',
]
