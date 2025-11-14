"""
DataFrame module for nitro-pandas.

This module provides a pandas-like DataFrame wrapper around Polars DataFrames.
The wrapper maintains pandas-like API while using Polars as the backend for
high-performance data operations.

Key features:
- Pandas-like API for familiar syntax
- Polars backend for optimized performance
- Automatic fallback to pandas for unimplemented methods
- Support for pandas-style boolean indexing and loc/iloc
"""

import polars as pl
import pandas as pd
import re
import numpy as np


class GroupBy:
    """
    Wrapper for Polars GroupBy operations with pandas-like API.
    
    This class enables pandas-style groupby operations like:
    - df.groupby("col").mean()
    - df.groupby("col")["other_col"].mean()
    - df.groupby("col").agg({"col": "mean"})
    
    All operations use Polars backend for performance.
    """
    
    def __init__(self, gb):
        """
        Initialize GroupBy wrapper.
        
        Args:
            gb: Polars GroupBy object
        """
        self._gb = gb
    
    def __getitem__(self, key):
        """
        Support for pandas-like column selection: df.groupby("col1")["col2"]
        
        Args:
            key: Column name (str) or list of column names
            
        Returns:
            GroupByColumn: Wrapper for column-specific groupby operations
            
        Raises:
            TypeError: If key is not str or list
        """
        if isinstance(key, str):
            return GroupByColumn(self._gb, key)
        elif isinstance(key, list):
            return GroupByColumn(self._gb, key)
        else:
            raise TypeError(f"GroupBy indices must be str or list, got {type(key)}")
    
    def mean(self):
        """Compute mean for all numeric columns in each group."""
        return DataFrame(self._gb.mean())
    
    def sum(self):
        """Compute sum for all numeric columns in each group."""
        return DataFrame(self._gb.sum())
    
    def min(self):
        """Compute minimum for all numeric columns in each group."""
        return DataFrame(self._gb.min())
    
    def max(self):
        """Compute maximum for all numeric columns in each group."""
        return DataFrame(self._gb.max())
    
    def count(self):
        """Count rows in each group."""
        return DataFrame(self._gb.count())
    
    def agg(self, *args, **kwargs):
        """
        Aggregate operations with pandas-like dictionary syntax.
        
        Supports both pandas-style dict aggregation and Polars expressions:
        - df.groupby("col").agg({"col": "mean"})  # pandas-style
        - df.groupby("col").agg(pl.col("col").mean())  # Polars-style
        
        Args:
            *args: Aggregation expressions or dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame: Aggregated results
            
        Raises:
            ValueError: If unsupported aggregation function is used
        """
        import polars as pl
        
        # Convert pandas-style dict to Polars expressions
        if len(args) == 1 and isinstance(args[0], dict):
            agg_dict = args[0]
            pl_expressions = []
            for col, func in agg_dict.items():
                if isinstance(func, str):
                    # Map string function names to Polars expressions
                    if func == 'mean':
                        pl_expressions.append(pl.col(col).mean())
                    elif func == 'sum':
                        pl_expressions.append(pl.col(col).sum())
                    elif func == 'min':
                        pl_expressions.append(pl.col(col).min())
                    elif func == 'max':
                        pl_expressions.append(pl.col(col).max())
                    elif func == 'count':
                        pl_expressions.append(pl.col(col).count())
                    else:
                        raise ValueError(
                            f"Unsupported aggregation function '{func}'. "
                            f"Use 'mean', 'sum', 'min', 'max', or 'count'"
                        )
                elif callable(func):
                    raise ValueError(
                        f"Lambda functions are not supported in groupby.agg(). "
                        f"Use string functions like 'mean', 'sum', 'min', 'max', or 'count'"
                    )
                else:
                    raise ValueError(f"Unsupported function type: {type(func)}")
            return DataFrame(self._gb.agg(pl_expressions))
        
        # Pass through Polars expressions directly
        return DataFrame(self._gb.agg(*args, **kwargs))


class GroupByColumn:
    """
    Wrapper for column-specific groupby operations.
    
    Enables pandas-style syntax: df.groupby("col1")["col2"].mean()
    All operations use Polars backend for performance.
    """
    
    def __init__(self, gb, column):
        """
        Initialize GroupByColumn wrapper.
        
        Args:
            gb: Polars GroupBy object
            column: Column name (str) or list of column names
        """
        import polars as pl
        self._gb = gb
        self._column = column
    
    def mean(self):
        """Compute mean for selected column(s) in each group."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).mean())
            return DataFrame(result)
        else:
            result = self._gb.agg([pl.col(col).mean().alias(col) for col in self._column])
            return DataFrame(result)
    
    def sum(self):
        """Compute sum for selected column(s) in each group."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).sum())
            return DataFrame(result)
        else:
            result = self._gb.agg([pl.col(col).sum().alias(col) for col in self._column])
            return DataFrame(result)
    
    def min(self):
        """Compute minimum for selected column(s) in each group."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).min())
            return DataFrame(result)
        else:
            result = self._gb.agg([pl.col(col).min().alias(col) for col in self._column])
            return DataFrame(result)
    
    def max(self):
        """Compute maximum for selected column(s) in each group."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).max())
            return DataFrame(result)
        else:
            result = self._gb.agg([pl.col(col).max().alias(col) for col in self._column])
            return DataFrame(result)
    
    def count(self):
        """Count rows for selected column(s) in each group."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).count().alias(self._column))
            return DataFrame(result)
        else:
            result = self._gb.agg([pl.col(col).count().alias(col) for col in self._column])
            return DataFrame(result)


class DataFrame:
    """
    Pandas-like DataFrame wrapper around Polars DataFrame.
    
    This class provides a pandas-compatible API while using Polars as the
    backend for high-performance data operations. It maintains pandas-like
    syntax for familiar usage while leveraging Polars' optimized engine.
    
    Key design principles:
    - Pandas-like API for user-facing operations
    - Polars backend for all data processing
    - Automatic fallback to pandas for unimplemented methods
    - Returns nitro-pandas DataFrame objects for chaining
    
    Example:
        >>> import nitro_pandas as npd
        >>> df = npd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> df.loc[df['a'] > 1]  # Pandas-like boolean indexing
        >>> df.groupby('a')['b'].mean()  # Pandas-like groupby
    """
    
    def __init__(self, data=None, pl_df: pl.DataFrame = None):
        """
        Initialize DataFrame from various data sources.
        
        Args:
            data: Data source (dict, list, Polars DataFrame, etc.)
            pl_df: Direct Polars DataFrame (for internal use)
            
        Examples:
            >>> df = DataFrame({'a': [1, 2, 3]})  # From dict
            >>> df = DataFrame(pl.DataFrame({'a': [1, 2, 3]}))  # From Polars
            >>> df = DataFrame()  # Empty DataFrame
        """
        import polars as pl
        
        if pl_df is not None:
            # Direct Polars DataFrame provided (internal use)
            self._df = pl_df
        elif data is not None:
            if isinstance(data, pl.DataFrame):
                # Already a Polars DataFrame
                self._df = data
            else:
                # Create Polars DataFrame from data (dict, list, etc.)
                self._df = pl.DataFrame(data)
        else:
            # Empty DataFrame
            self._df = pl.DataFrame()

    def __getattr__(self, name: str):
        """
        Automatic fallback to pandas for unimplemented methods.
        
        This enables access to all pandas methods not explicitly implemented
        in nitro-pandas. The result is returned as-is (pandas DataFrame/Series).
        
        Args:
            name: Method or attribute name
            
        Returns:
            Method or attribute from pandas DataFrame
            
        Raises:
            AttributeError: If attribute doesn't exist in pandas either
        """
        import pandas as pd

        # Convert to pandas for fallback
        pdf = self._df.to_pandas()
        if not hasattr(pdf, name):
            raise AttributeError(f"'DataFrame' object has no attribute '{name}'")

        attr = getattr(pdf, name)
        if callable(attr):
            # Wrap callable methods to return pandas objects directly
            def _pandas_fallback(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    return result
                return result
            return _pandas_fallback

        # Return non-callable attributes directly
        if isinstance(attr, (pd.DataFrame, pd.Series)):
            return attr
        return attr

    def __repr__(self):
        """
        String representation of DataFrame (pandas-like display).
        
        Returns a readable string representation of the DataFrame,
        similar to pandas' display format.
        
        Returns:
            str: String representation of DataFrame
        """
        # Use Polars' built-in string representation which is already nice
        return self._df.__repr__()
    
    def __str__(self):
        """
        String representation of DataFrame.
        
        Returns:
            str: String representation of DataFrame
        """
        return self._df.__str__()

    def query(self, expr: str):
        """
        Filter DataFrame using a query expression (pandas-like).
        
        Converts pandas-style query strings to Polars expressions.
        Supports 'and'/'or' operators and column references.
        
        Args:
            expr: Query string (e.g., "col1 > 2 and col2 == 'A'")
            
        Returns:
            DataFrame: Filtered DataFrame
            
        Example:
            >>> df.query("id > 2 and name == 'Bob'")
        """
        # Replace column names with pl.col() expressions
        def repl(match):
            col = match.group(0)
            return f'pl.col("{col}")'
        
        # Build regex pattern to match column names
        pattern = r'\b(' + '|'.join(map(re.escape, self.columns)) + r')\b'
        expr_polars = re.sub(pattern, repl, expr)
        
        # Convert pandas-style operators to Polars syntax
        expr_polars = re.sub(r'\s+and\s+', ' & ', expr_polars)
        expr_polars = re.sub(r'\s+or\s+', ' | ', expr_polars) 
        expr_polars = re.sub(r'\band\b', '&', expr_polars)
        expr_polars = re.sub(r'\bor\b', '|', expr_polars)
        
        # Add parentheses around comparison expressions for proper evaluation
        expr_polars = re.sub(
            r'(pl\.col\([^)]+\)\s*[<>=!]=?\s*[^&|()]+)',
            lambda m: '(' + m.group(1).strip() + ')',
            expr_polars
        )
        
        # Evaluate expression and filter
        import polars as pl
        mask_expr = eval(expr_polars, {"pl": pl})
        return DataFrame(self._df.filter(mask_expr))

    def __gt__(self, other):
        """
        Greater than comparison operator: df > value
        
        Compares all numeric columns with a value and returns a pandas
        DataFrame boolean mask. This enables pandas-style boolean indexing
        like df.loc[df > 2].
        
        Args:
            other: Value to compare against
            
        Returns:
            pandas.DataFrame: Boolean DataFrame with same shape
            
        Note:
            Only numeric columns are compared. Non-numeric columns
            are set to False in the result.
        """
        import polars as pl
        import pandas as pd
        
        # Identify numeric columns
        numeric_cols = [
            col for col in self._df.columns
            if self._df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]
        
        if len(numeric_cols) == 0:
            # No numeric columns, return all False
            return pd.DataFrame(
                False,
                index=range(self._df.height),
                columns=self._df.columns
            )
        
        # Build Polars expressions for comparison
        result_exprs = []
        for col in self._df.columns:
            if col in numeric_cols:
                result_exprs.append(pl.col(col) > other)
            else:
                result_exprs.append(pl.lit(False).alias(col))
        
        # Execute comparison with Polars and convert to pandas
        result_pl = self._df.select(result_exprs)
        return result_pl.to_pandas()

    def __lt__(self, other):
        """Less than comparison: df < value"""
        import polars as pl
        import pandas as pd
        numeric_cols = [
            col for col in self._df.columns
            if self._df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]
        if len(numeric_cols) == 0:
            return pd.DataFrame(False, index=range(self._df.height), columns=self._df.columns)
        result_exprs = []
        for col in self._df.columns:
            if col in numeric_cols:
                result_exprs.append(pl.col(col) < other)
            else:
                result_exprs.append(pl.lit(False).alias(col))
        result_pl = self._df.select(result_exprs)
        return result_pl.to_pandas()

    def __ge__(self, other):
        """Greater than or equal comparison: df >= value"""
        import polars as pl
        import pandas as pd
        numeric_cols = [
            col for col in self._df.columns
            if self._df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]
        if len(numeric_cols) == 0:
            return pd.DataFrame(False, index=range(self._df.height), columns=self._df.columns)
        result_exprs = []
        for col in self._df.columns:
            if col in numeric_cols:
                result_exprs.append(pl.col(col) >= other)
            else:
                result_exprs.append(pl.lit(False).alias(col))
        result_pl = self._df.select(result_exprs)
        return result_pl.to_pandas()

    def __le__(self, other):
        """Less than or equal comparison: df <= value"""
        import polars as pl
        import pandas as pd
        numeric_cols = [
            col for col in self._df.columns
            if self._df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]
        if len(numeric_cols) == 0:
            return pd.DataFrame(False, index=range(self._df.height), columns=self._df.columns)
        result_exprs = []
        for col in self._df.columns:
            if col in numeric_cols:
                result_exprs.append(pl.col(col) <= other)
            else:
                result_exprs.append(pl.lit(False).alias(col))
        result_pl = self._df.select(result_exprs)
        return result_pl.to_pandas()

    def __eq__(self, other):
        """Equal comparison: df == value"""
        import polars as pl
        import pandas as pd
        numeric_cols = [
            col for col in self._df.columns
            if self._df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]
        if len(numeric_cols) == 0:
            return pd.DataFrame(False, index=range(self._df.height), columns=self._df.columns)
        result_exprs = []
        for col in self._df.columns:
            if col in numeric_cols:
                result_exprs.append(pl.col(col) == other)
            else:
                result_exprs.append(pl.lit(False).alias(col))
        result_pl = self._df.select(result_exprs)
        return result_pl.to_pandas()

    def __ne__(self, other):
        """Not equal comparison: df != value"""
        import polars as pl
        import pandas as pd
        numeric_cols = [
            col for col in self._df.columns
            if self._df[col].dtype in [
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ]
        ]
        if len(numeric_cols) == 0:
            return pd.DataFrame(False, index=range(self._df.height), columns=self._df.columns)
        result_exprs = []
        for col in self._df.columns:
            if col in numeric_cols:
                result_exprs.append(pl.col(col) != other)
            else:
                result_exprs.append(pl.lit(False).alias(col))
        result_pl = self._df.select(result_exprs)
        return result_pl.to_pandas()

    def __getitem__(self, key):
        """
        Indexing operator for column selection and boolean filtering.
        
        Supports multiple indexing patterns:
        - df['col']: Single column (returns pandas Series)
        - df[['col1', 'col2']]: Multiple columns (returns DataFrame)
        - df[mask]: Boolean filtering (returns DataFrame)
        
        Args:
            key: Column name, list of columns, or boolean mask
            
        Returns:
            pandas.Series: For single column selection
            DataFrame: For multiple columns or boolean filtering
            
        Example:
            >>> df['a']  # Returns pandas Series
            >>> df[['a', 'b']]  # Returns DataFrame
            >>> df[df['a'] > 2]  # Boolean filtering
        """
        import polars as pl
        import numpy as np
        import pandas as pd
        
        # Handle pandas Series boolean mask (from df['col'] > value)
        if isinstance(key, pd.Series):
            mask_values = key.tolist()
            mask = pl.Series("", mask_values).cast(pl.Boolean)
            return DataFrame(self._df.filter(mask))
        
        # Handle other boolean masks (Polars Series, numpy array, list)
        if (isinstance(key, pl.Series) and key.dtype == pl.Boolean) or \
           (isinstance(key, np.ndarray) and key.dtype == bool) or \
           (isinstance(key, list) and all(isinstance(x, (bool, np.bool_)) for x in key)):
            mask = pl.Series("", key).cast(pl.Boolean)
            return DataFrame(self._df.filter(mask))
        
        # Handle column selection: df[['col1', 'col2']]
        if isinstance(key, list) and all(isinstance(x, str) for x in key):
            return DataFrame(self._df.select(key))
        
        # Handle single column: df['col'] (returns pandas Series for pandas expressions)
        if isinstance(key, str):
            import pandas as pd
            pl_series = self._df[key]
            # Convert to pandas Series to enable pandas-style expressions
            return pl_series.to_pandas()
        
        # Fallback to Polars indexing (slices, tuples, etc.)
        result = self._df[key]
        if isinstance(result, pl.DataFrame):
            return DataFrame(result)
        return result

    def __setitem__(self, key, value):
        """
        Assignment operator for adding or modifying columns (pandas-like).
        
        Supports:
        - df['new_col'] = scalar: Add column with constant value
        - df['new_col'] = list: Add column from list/array
        - df['new_col'] = pl.Expr: Add column from Polars expression
        - df['existing_col'] = value: Modify existing column
        
        Args:
            key: Column name (str)
            value: Column value (scalar, list, Polars expression, or pandas Series)
            
        Example:
            >>> df['new_col'] = 10  # Add column with constant
            >>> df['doubled'] = df['value'] * 2  # Add column from expression
            >>> df['scores'] = [100, 200, 300]  # Add column from list
        """
        import polars as pl
        import pandas as pd
        
        if not isinstance(key, str):
            raise TypeError(f"Column assignment requires string key, got {type(key)}")
        
        # Handle Polars expression
        if isinstance(value, pl.Expr):
            self._df = self._df.with_columns(value.alias(key))
        # Handle pandas Series
        elif isinstance(value, pd.Series):
            self._df = self._df.with_columns(pl.Series(key, value.tolist()))
        # Handle list/array
        elif isinstance(value, (list, tuple)):
            self._df = self._df.with_columns(pl.Series(key, value))
        # Handle scalar value
        else:
            self._df = self._df.with_columns(pl.lit(value).alias(key))

    @property
    def columns(self):
        """Return column labels as a list."""
        return self._df.columns

    @columns.setter
    def columns(self, new_columns):
        """Set column labels."""
        self._df.columns = new_columns
    
    def head(self, n: int = 5) -> "DataFrame":
        """Return the first n rows."""
        return DataFrame(self._df.head(n))
    
    def tail(self, n: int = 5) -> "DataFrame":
        """Return the last n rows."""
        return DataFrame(self._df.tail(n))
    
    @property
    def shape(self) -> tuple:
        """Return a tuple representing the dimensionality (rows, columns)."""
        return (self._df.height, self._df.width)
    
    def to_pandas(self):
        """Convert to pandas DataFrame."""
        return self._df.to_pandas()

    def to_csv(self, path, **kwargs):
        """Write DataFrame to a CSV file using Polars backend."""
        self._df.write_csv(path, **kwargs)

    def to_parquet(self, path, **kwargs):
        """Write DataFrame to a Parquet file using Polars backend."""
        self._df.write_parquet(path, **kwargs)

    def to_json(self, path, **kwargs):
        """Write DataFrame to a JSON file using Polars backend."""
        self._df.write_json(path, **kwargs)

    def to_excel(self, path, **kwargs):
        """
        Write DataFrame to an Excel file using pandas (fallback).
        
        Note: Polars doesn't have native Excel writing support,
        so we use pandas as a fallback.
        """
        pdf = self._df.to_pandas()
        pdf.to_excel(path, index=False, **kwargs)

    def groupby(self, by):
        """
        Group DataFrame by one or more columns.
        
        Args:
            by: Column name(s) to group by
            
        Returns:
            GroupBy: GroupBy object for aggregation operations
        """
        return GroupBy(self._df.group_by(by)) 

    @property
    def loc(self):
        """Label-based indexing (pandas-like)."""
        return LocIndexer(self)

    @property
    def iloc(self):
        """Integer position-based indexing (pandas-like)."""
        return ILocIndexer(self)

    def sort_values(self, by, ascending: bool = True, na_position: str = "last"):
        """
        Sort DataFrame by one or more columns.
        
        Args:
            by: Column name(s) to sort by
            ascending: Sort in ascending order
            na_position: Position of null values ('first' or 'last')
            
        Returns:
            DataFrame: Sorted DataFrame
        """
        import polars as pl
        cols = by if isinstance(by, list) else [by]
        nulls_last = True if na_position == "last" else False
        out = self._df.sort(by=cols, descending=not ascending, nulls_last=nulls_last)
        return DataFrame(out)

    def rename(self, columns: dict | None = None):
        """
        Rename columns.
        
        Args:
            columns: Dictionary mapping old names to new names
            
        Returns:
            DataFrame: DataFrame with renamed columns
        """
        if not columns:
            return DataFrame(self._df)
        out = self._df.rename(columns)
        return DataFrame(out)

    def drop(self, labels, axis=0):
        """
        Drop rows or columns.
        
        Args:
            labels: Labels to drop (row indices or column names)
            axis: 0 for rows, 1 for columns
            
        Returns:
            DataFrame: DataFrame with dropped rows/columns
            
        Raises:
            ValueError: If axis is not 0 or 1
        """
        import polars as pl
        
        if labels is None:
            return DataFrame(self._df)
        
        if axis == 0:
            # Drop rows by index
            if isinstance(labels, (int, list)):
                if isinstance(labels, int):
                    labels = [labels]
                # Create mask to exclude specified rows
                mask = ~pl.int_range(0, self._df.height).is_in(labels)
                out = self._df.filter(mask)
                return DataFrame(out)
            elif isinstance(labels, slice):
                # Handle slice notation
                start = labels.start if labels.start is not None else 0
                stop = labels.stop if labels.stop is not None else self._df.height
                step = labels.step if labels.step is not None else 1
                indices_to_remove = list(range(start, stop, step))
                mask = ~pl.int_range(0, self._df.height).is_in(indices_to_remove)
                out = self._df.filter(mask)
                return DataFrame(out)
        
        elif axis == 1:
            # Drop columns
            if isinstance(labels, str):
                labels = [labels]
            out = self._df.drop(labels)
            return DataFrame(out)
        
        raise ValueError(f"axis must be 0 (rows) or 1 (columns), got {axis}")

    def astype(self, dtype):
        """
        Convert column types using pandas-like type names.
        
        Supports both pandas-style type names (str, int, float) and
        Polars types. Converts pandas types to Polars internally.
        
        Args:
            dtype: Type mapping (dict) or single type for all columns
            
        Returns:
            DataFrame: DataFrame with converted types
            
        Example:
            >>> df.astype({'id': 'int64', 'name': 'str'})
            >>> df.astype('float')  # Convert all columns
        """
        import polars as pl
        
        def _pandas_to_polars_type(pandas_type):
            """Convert pandas type representation to Polars type."""
            if isinstance(pandas_type, str):
                type_map = {
                    'str': pl.String,
                    'string': pl.String,
                    'int': pl.Int64,
                    'int64': pl.Int64,
                    'int32': pl.Int32,
                    'float': pl.Float64,
                    'float64': pl.Float64,
                    'float32': pl.Float32,
                    'bool': pl.Boolean,
                    'boolean': pl.Boolean,
                    'datetime64': pl.Datetime,
                    'datetime': pl.Datetime,
                    'date': pl.Date,
                }
                return type_map.get(pandas_type.lower(), pl.String)
            elif pandas_type == str:
                return pl.String
            elif pandas_type == int:
                return pl.Int64
            elif pandas_type == float:
                return pl.Float64
            elif pandas_type == bool:
                return pl.Boolean
            else:
                # Already a Polars type
                return pandas_type
        
        if isinstance(dtype, dict):
            # Per-column type mapping
            out = self._df
            for col, t in dtype.items():
                pl_type = _pandas_to_polars_type(t)
                out = out.with_columns(pl.col(col).cast(pl_type))
            return DataFrame(out)
        else:
            # Single type for all columns
            pl_type = _pandas_to_polars_type(dtype)
            out = self._df.select([pl.col(c).cast(pl_type).alias(c) for c in self._df.columns])
            return DataFrame(out)

    def fillna(self, value):
        """
        Fill null values.
        
        Args:
            value: Fill value (scalar or dict mapping columns to values)
            
        Returns:
            DataFrame: DataFrame with filled null values
        """
        import polars as pl
        if isinstance(value, dict):
            # Per-column fill values
            out = self._df
            for col, v in value.items():
                out = out.with_columns(pl.col(col).fill_null(v))
            return DataFrame(out)
        else:
            # Single fill value for all columns
            out = self._df.select([pl.col(c).fill_null(value).alias(c) for c in self._df.columns])
            return DataFrame(out)

    def drop_duplicates(self, subset: list[str] | None = None, keep: str = "first"):
        """
        Remove duplicate rows.
        
        Args:
            subset: Columns to consider for duplicates (None = all columns)
            keep: Which duplicates to keep ('first', 'last', or False for none)
            
        Returns:
            DataFrame: DataFrame with duplicates removed
        """
        keep_map = {"first": "first", "last": "last", False: "none", None: "first"}
        out = self._df.unique(subset=subset, keep=keep_map.get(keep, "first"))
        return DataFrame(out)

    def value_counts(self, column: str, sort: bool = True, ascending: bool = False):
        """
        Count unique values in a column.
        
        Args:
            column: Column name to count
            sort: Whether to sort results
            ascending: Sort in ascending order
            
        Returns:
            DataFrame: DataFrame with value counts
        """
        import polars as pl
        out = self._df.group_by(column).agg(pl.count().alias("count"))
        if sort:
            out = out.sort("count", descending=not ascending)
        return DataFrame(out)

    def reset_index(self, drop: bool = True, name: str = "index"):
        """
        Reset index (add row numbers as column).
        
        Args:
            drop: If True, don't add index column
            name: Name for index column if not dropped
            
        Returns:
            DataFrame: DataFrame with reset index
        """
        import polars as pl
        if drop:
            return DataFrame(self._df)
        out = self._df.with_row_count(name)
        return DataFrame(out)

    def merge(self, right: "DataFrame", how: str = "inner", on: str | list[str] | None = None,
              left_on: str | list[str] | None = None, right_on: str | list[str] | None = None, suffixes=("_x","_y")):
        """
        Merge two DataFrames (pandas-like join).
        
        Args:
            right: Right DataFrame to merge
            how: Join type ('inner', 'left', 'right', 'outer', 'cross')
            on: Column name(s) to join on (if same in both)
            left_on: Column name(s) in left DataFrame
            right_on: Column name(s) in right DataFrame
            suffixes: Suffixes for overlapping columns
            
        Returns:
            DataFrame: Merged DataFrame
        """
        how_map = {"inner":"inner", "left":"left", "right":"right", "outer":"outer", "cross":"cross"}
        if on is not None:
            left_on = on
            right_on = on
        out = self._df.join(
            right._df,
            left_on=left_on,
            right_on=right_on,
            how=how_map.get(how, "inner"),
            suffix=suffixes[1]
        )
        return DataFrame(out)

    @staticmethod
    def concat(dfs: list["DataFrame"], axis: int = 0):
        """
        Concatenate multiple DataFrames (deprecated, use npd.concat() instead).
        
        This method is kept for backward compatibility. Prefer using:
        >>> import nitro_pandas as npd
        >>> npd.concat([df1, df2])
        
        Args:
            dfs: List of DataFrames to concatenate
            axis: 0 for vertical (row-wise), 1 for horizontal (column-wise)
            
        Returns:
            DataFrame: Concatenated DataFrame
        """
        # Import here to avoid circular dependency
        from . import concat as module_concat
        return module_concat(dfs, axis=axis)

    def isna(self):
        """Return boolean DataFrame indicating null values."""
        import polars as pl
        out = self._df.select([pl.col(c).is_null().alias(c) for c in self._df.columns])
        return DataFrame(out)

    def notna(self):
        """Return boolean DataFrame indicating non-null values."""
        import polars as pl
        out = self._df.select([pl.col(c).is_not_null().alias(c) for c in self._df.columns])
        return DataFrame(out)


class LocIndexer:
    """
    Label-based indexer for DataFrame.loc[] (pandas-like).
    
    Supports various indexing patterns:
    - df.loc[mask]: Boolean filtering
    - df.loc[2:5]: Slice selection
    - df.loc[mask, 'col']: Boolean filtering with column selection
    - df.loc[df > 2]: DataFrame boolean mask
    
    All operations use Polars backend for filtering.
    """
    
    def __init__(self, df):
        """Initialize LocIndexer with DataFrame reference."""
        self.df = df

    def __getitem__(self, key):
        """
        Label-based indexing with pandas-like syntax.
        
        Args:
            key: Indexing key (mask, slice, int, list, or tuple)
            
        Returns:
            DataFrame, Series, or scalar: Depending on selection
            
        Raises:
            ValueError: If mask length doesn't match DataFrame height
            NotImplementedError: If indexing type is not supported
        """
        import polars as pl
        import pandas as pd

        # Parse key as (rows, cols) tuple or just rows
        if isinstance(key, tuple):
            rows, cols = key
        else:
            rows = key
            cols = None

        # Process column selection
        pl_cols = None
        if cols is not None:
            if isinstance(cols, slice):
                pl_cols = self.df._df.columns[cols]
            elif isinstance(cols, str):
                pl_cols = [cols]
            elif isinstance(cols, list):
                pl_cols = cols
            else:
                raise NotImplementedError("loc: unsupported column selection type")

        # Handle pandas Series boolean mask (from df['col'] > value)
        import pandas as pd
        import numpy as np
        if isinstance(rows, pd.Series):
            mask_values = rows.tolist()
            mask = pl.Series("", mask_values).cast(pl.Boolean)
            if len(mask) != self.df._df.height:
                raise ValueError(
                    f"Mask length {len(mask)} does not match DataFrame height {self.df._df.height}"
                )
            filtered = self.df._df.filter(mask)
            result = filtered if pl_cols is None else filtered.select(pl_cols)
        
        # Handle pandas DataFrame boolean mask (from df > value)
        elif isinstance(rows, pd.DataFrame):
            # Convert DataFrame mask to row mask using any() per row
            # This filters rows where at least one column is True
            mask_values = rows.any(axis=1).tolist()
            mask = pl.Series("", mask_values).cast(pl.Boolean)
            if len(mask) != self.df._df.height:
                raise ValueError(
                    f"Mask length {len(mask)} does not match DataFrame height {self.df._df.height}"
                )
            filtered = self.df._df.filter(mask)
            result = filtered if pl_cols is None else filtered.select(pl_cols)
        
        # Handle Polars Series boolean mask (for compatibility)
        elif isinstance(rows, pl.Series) and rows.dtype == pl.Boolean:
            if len(rows) != self.df._df.height:
                raise ValueError(
                    f"Mask length {len(rows)} does not match DataFrame height {self.df._df.height}"
                )
            filtered = self.df._df.filter(rows)
            result = filtered if pl_cols is None else filtered.select(pl_cols)

        # Handle slice notation: df.loc[2:5]
        elif isinstance(rows, slice):
            start = rows.start if rows.start is not None else 0
            stop = rows.stop if rows.stop is not None else self.df._df.height - 1
            step = rows.step if rows.step is not None else 1
            indices = list(range(start, stop + 1, step))
            result = self.df._df[indices] if pl_cols is None else self.df._df[indices].select(pl_cols)

        # Handle integer or list of indices
        elif isinstance(rows, (int, list)):
            result = self.df._df[rows] if pl_cols is None else self.df._df[rows].select(pl_cols)

        # Fallback: try to convert to boolean mask
        else:
            try:
                if isinstance(rows, np.ndarray) and rows.dtype == bool:
                    mask = pl.Series("", rows.tolist()).cast(pl.Boolean)
                elif isinstance(rows, list) and all(isinstance(x, (bool, np.bool_)) for x in rows):
                    mask = pl.Series("", rows).cast(pl.Boolean)
                elif hasattr(rows, 'to_list'):
                    mask = pl.Series("", rows.to_list()).cast(pl.Boolean)
                else:
                    mask = pl.Series("", list(rows)).cast(pl.Boolean)
                
                if len(mask) != self.df._df.height:
                    raise ValueError(
                        f"Mask length {len(mask)} does not match DataFrame height {self.df._df.height}"
                    )
                filtered = self.df._df.filter(mask)
                result = filtered if pl_cols is None else filtered.select(pl_cols)
            except Exception as e:
                raise ValueError(
                    f"loc: unsupported row selection type: {type(rows)}. Error: {str(e)}"
                )

        # Process result and return appropriate type
        if isinstance(result, pl.DataFrame):
            pdf = result.to_pandas()
            # Single value: return scalar
            if pdf.shape == (1, 1):
                return pdf.iloc[0, 0]
            # Single column: return Series or scalar
            if pl_cols is not None and isinstance(pl_cols, list) and len(pl_cols) == 1:
                s = pl.from_pandas(pdf.iloc[:, 0].reset_index(drop=True))
                return s if len(s) != 1 else s.item()
            return DataFrame(result)
        return result


class ILocIndexer:
    """
    Integer position-based indexer for DataFrame.iloc[] (pandas-like).
    
    Supports integer-based indexing:
    - df.iloc[0]: First row
    - df.iloc[0:5]: Slice of rows
    - df.iloc[0, 0]: Single value
    - df.iloc[0:5, 0:2]: Row and column slices
    """
    
    def __init__(self, df):
        """Initialize ILocIndexer with DataFrame reference."""
        self.df = df
    
    def __getitem__(self, key):
        """
        Integer position-based indexing.
        
        Args:
            key: Integer, slice, list, or tuple (rows, cols)
            
        Returns:
            DataFrame, Series, or scalar: Depending on selection
            
        Raises:
            NotImplementedError: If indexing type is not supported
        """
        # Parse key as (rows, cols) tuple or just rows
        if isinstance(key, tuple):
            rows, cols = key
        else:
            rows, cols = key, slice(None)
        
        # Process row selection by position
        if isinstance(rows, int):
            pl_rows = [rows]
        elif isinstance(rows, slice):
            pl_rows = list(range(*rows.indices(self.df._df.height)))
        elif isinstance(rows, list):
            pl_rows = rows
        else:
            raise NotImplementedError("iloc: unsupported row selection type")
        
        # Process column selection by position
        if isinstance(cols, int):
            pl_cols = [self.df._df.columns[cols]]
        elif isinstance(cols, slice):
            pl_cols = self.df._df.columns[cols]
        elif isinstance(cols, list):
            pl_cols = [self.df._df.columns[i] if isinstance(i, int) else i for i in cols]
        else:
            raise NotImplementedError("iloc: unsupported column selection type")
        
        # Use pandas for advanced indexing (handles edge cases better)
        pdf = self.df._df.to_pandas()
        result_pd = pdf.iloc[pl_rows][pl_cols]
        import pandas as pd
        
        # Return appropriate type based on result shape
        if isinstance(result_pd, pd.Series) and result_pd.shape == ():
            return result_pd.item()
        if isinstance(result_pd, pd.Series):
            if len(result_pd) == 1:
                return result_pd.item()
            return pl.from_pandas(result_pd.reset_index(drop=True))
        if isinstance(result_pd, pd.DataFrame) and result_pd.shape[1] == 1:
            s = pl.from_pandas(result_pd.iloc[:, 0].reset_index(drop=True))
            if len(s) == 1:
                return s.item()
            return s
        return DataFrame(pl.from_pandas(result_pd.reset_index(drop=True)))
