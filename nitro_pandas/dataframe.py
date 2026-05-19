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
import warnings

class PandasFallbackWarning(UserWarning):
    """Warning emitted when nitro-pandas falls back to pandas.

    Silence with:
        import warnings, nitro_pandas as npd
        warnings.filterwarnings("ignore", category=npd.PandasFallbackWarning)
    """


class _StringAccessor:
    """Translates pandas str.contains API to Polars regex."""

    def __init__(self, pl_str):
        self._str = pl_str

    def contains(self, pat, case=True, na=False, regex=True):
        if not case:
            pat = f"(?i){pat}"
        return self._str.contains(pat, literal=not regex)

    def __getattr__(self, name):
        return getattr(self._str, name)


class Series:
    """Thin Polars-backed wrapper with a pandas-like API.

    Keeps column data in Polars so comparisons, fillna, and str.contains
    all stay in Polars land — no Python list round-trips.
    """

    def __init__(self, pl_series: pl.Series):
        self._series = pl_series

    # ── string accessor ───────────────────────────────────────────────────
    @property
    def str(self):
        return _StringAccessor(self._series.str)

    # ── null handling ─────────────────────────────────────────────────────
    def fillna(self, value):
        return Series(self._series.fill_null(value))

    # ── type casting ──────────────────────────────────────────────────────
    def astype(self, dtype):
        dtype_map = {
            "int64": pl.Int64, "int32": pl.Int32,
            "float64": pl.Float64, "float32": pl.Float32,
            "str": pl.Utf8, "bool": pl.Boolean,
            int: pl.Int64, float: pl.Float64, str: pl.Utf8, bool: pl.Boolean,
        }
        return Series(self._series.cast(dtype_map.get(dtype, dtype)))

    # ── value counts ──────────────────────────────────────────────────────
    def value_counts(self, sort=True, ascending=False):
        result = self._series.value_counts(sort=sort, parallel=True)
        if sort and not ascending:
            result = result.sort("count", descending=True)
        return DataFrame(result)

    # ── comparisons → pl.Series (usable directly as boolean mask) ────────
    def __gt__(self, other):  return self._series.__gt__(other)
    def __lt__(self, other):  return self._series.__lt__(other)
    def __ge__(self, other):  return self._series.__ge__(other)
    def __le__(self, other):  return self._series.__le__(other)
    def __eq__(self, other):  return self._series.__eq__(other)
    def __ne__(self, other):  return self._series.__ne__(other)

    # ── arithmetic → pl.Series ────────────────────────────────────────────
    def __add__(self, other):       return self._series + other
    def __radd__(self, other):      return other + self._series
    def __sub__(self, other):       return self._series - other
    def __rsub__(self, other):      return other - self._series
    def __mul__(self, other):       return self._series * other
    def __rmul__(self, other):      return other * self._series
    def __truediv__(self, other):   return self._series / other
    def __floordiv__(self, other):  return self._series // other
    def __mod__(self, other):       return self._series % other

    # ── boolean combination ───────────────────────────────────────────────
    def __and__(self, other):
        return self._series & (other._series if isinstance(other, Series) else other)

    def __or__(self, other):
        return self._series | (other._series if isinstance(other, Series) else other)

    def __invert__(self):
        return ~self._series

    # ── sizing / iteration ────────────────────────────────────────────────
    def __len__(self):   return len(self._series)
    def __iter__(self):  return iter(self._series)
    def __repr__(self):  return self._series.__repr__()
    def __str__(self):   return self._series.__str__()

    # ── conversion ────────────────────────────────────────────────────────
    def to_pandas(self):  return self._series.to_pandas()
    def to_list(self):    return self._series.to_list()
    tolist = to_list

    # ── transparent fallback: pl.Series first, then pd.Series ────────────
    def __getattr__(self, name):
        if hasattr(self._series, name):
            return getattr(self._series, name)
        pd_series = self._series.to_pandas()
        if hasattr(pd_series, name):
            warnings.warn(
                f"[nitro-pandas] Series.'{name}' is not natively available — pandas fallback activated.",
                PandasFallbackWarning,
                stacklevel=2,
            )
            return getattr(pd_series, name)
        raise AttributeError(f"'Series' object has no attribute '{name}'")


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
        # Buffer for pending column assignments — flushed as a single with_columns()
        self._pending_cols = {}

    def _flush_pending(self, read_col: str | None = None) -> None:
        """Flush buffered column assignments as a single with_columns() call.

        Args:
            read_col: When set, only flush if that specific column has a pending
                      write (smart flush for df["col"] access). None = flush all.
        """
        if not self._pending_cols:
            return
        if read_col is not None and read_col not in self._pending_cols:
            return  # Reading an unmodified column — safe to skip
        exprs = []
        for name, val in self._pending_cols.items():
            if isinstance(val, Series):
                exprs.append(val._series.alias(name))
            elif isinstance(val, pl.Series):
                exprs.append(val.alias(name))
            elif isinstance(val, pl.Expr):
                exprs.append(val.alias(name))
            elif isinstance(val, pd.Series):
                exprs.append(pl.from_pandas(val).alias(name))
            elif isinstance(val, np.ndarray):
                exprs.append(pl.Series(name, val))
            elif isinstance(val, (list, tuple)):
                exprs.append(pl.Series(name, val))
            else:
                exprs.append(pl.lit(val).alias(name))
        self._df = self._df.with_columns(exprs)
        self._pending_cols = {}

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

        # Check existence on the pandas DataFrame class — no data conversion needed.
        if not hasattr(pd.DataFrame, name):
            raise AttributeError(f"'DataFrame' object has no attribute '{name}'")

        pd_attr = getattr(pd.DataFrame, name)
        if callable(pd_attr):
            # Defer to_pandas() until the method is actually called.
            def _pandas_fallback(*args, **kwargs):
                warnings.warn(
                    f"[nitro-pandas] '{name}' is not natively implemented — pandas fallback activated.",
                    PandasFallbackWarning,
                    stacklevel=2,
                )
                self._flush_pending()
                return getattr(self._df.to_pandas(), name)(*args, **kwargs)
            return _pandas_fallback

        # Non-callable attribute: convert now.
        self._flush_pending()
        warnings.warn(
            f"[nitro-pandas] '{name}' is not natively implemented — pandas fallback activated.",
            PandasFallbackWarning,
            stacklevel=2,
        )
        return getattr(self._df.to_pandas(), name)

    def __repr__(self):
        """
        String representation of DataFrame (pandas-like display).
        
        Returns a readable string representation of the DataFrame,
        similar to pandas' display format.
        
        Returns:
            str: String representation of DataFrame
        """
        self._flush_pending()
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

        Parses the expression safely via the AST — no eval() on user input.
        Only column comparisons and and/or combinations are allowed.
        Any attempt to call functions or import modules raises ValueError.

        Args:
            expr: Query string (e.g., "col1 > 2 and col2 == 'A'")

        Returns:
            DataFrame: Filtered DataFrame

        Example:
            >>> df.query("id > 2 and name == 'Bob'")
        """
        import ast as _ast

        columns = set(self._df.columns)

        def _build(node):
            # Boolean combinations: expr and expr / expr or expr
            if isinstance(node, _ast.BoolOp):
                parts = [_build(v) for v in node.values]
                result = parts[0]
                for part in parts[1:]:
                    result = result & part if isinstance(node.op, _ast.And) else result | part
                return result

            # Comparisons: col > value, col == 'x', ...
            if isinstance(node, _ast.Compare):
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise ValueError("Only simple comparisons are supported (one operator per expression).")
                if not isinstance(node.left, _ast.Name):
                    raise ValueError(f"Left side of comparison must be a column name.")
                col_name = node.left.id
                if col_name not in columns:
                    raise ValueError(f"Column '{col_name}' not found in DataFrame.")
                comparator = node.comparators[0]
                if not isinstance(comparator, _ast.Constant):
                    raise ValueError("Right side of comparison must be a literal value (number or string).")
                value = comparator.value
                col_expr = pl.col(col_name)
                op = node.ops[0]
                if isinstance(op, _ast.Gt):    return col_expr > value
                if isinstance(op, _ast.Lt):    return col_expr < value
                if isinstance(op, _ast.GtE):   return col_expr >= value
                if isinstance(op, _ast.LtE):   return col_expr <= value
                if isinstance(op, _ast.Eq):    return col_expr == value
                if isinstance(op, _ast.NotEq): return col_expr != value
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")

            raise ValueError(
                f"Unsupported expression type '{type(node).__name__}'. "
                f"Only column comparisons with and/or are allowed."
            )

        try:
            tree = _ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid query expression: {e}") from e

        self._flush_pending()
        polars_expr = _build(tree.body)
        return DataFrame(self._df.filter(polars_expr))

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
            # Optimize: use numpy values directly if available (faster than tolist())
            if hasattr(key, 'values') and isinstance(key.values, np.ndarray):
                mask = pl.Series("", key.values).cast(pl.Boolean)
            else:
                mask_values = key.tolist()
                mask = pl.Series("", mask_values).cast(pl.Boolean)
            return DataFrame(self._df.filter(mask))
        
        # Handle other boolean masks (our Series, Polars Series, numpy array, list)
        if isinstance(key, Series):
            self._flush_pending()
            return DataFrame(self._df.filter(key._series.cast(pl.Boolean)))
        if (isinstance(key, pl.Series) and key.dtype == pl.Boolean) or \
           (isinstance(key, np.ndarray) and key.dtype == bool) or \
           (isinstance(key, list) and all(isinstance(x, (bool, np.bool_)) for x in key)):
            self._flush_pending()
            mask = pl.Series("", key).cast(pl.Boolean)
            return DataFrame(self._df.filter(mask))

        # Handle column selection: df[['col1', 'col2']]
        if isinstance(key, list) and all(isinstance(x, str) for x in key):
            self._flush_pending()
            return DataFrame(self._df.select(key))

        # Handle single column: smart flush (only if this column has a pending write)
        if isinstance(key, str):
            self._flush_pending(read_col=key)
            return Series(self._df[key])
        
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
        
        # Buffer the assignment — will be flushed as a batched with_columns()
        # the next time the DataFrame is read. This avoids one with_columns()
        # call per assignment when multiple columns are added in sequence.
        self._pending_cols[key] = value

    @property
    def columns(self):
        """Return column labels as a list (includes pending columns)."""
        existing = self._df.columns
        pending = [k for k in self._pending_cols if k not in existing]
        return existing + pending

    @columns.setter
    def columns(self, new_columns):
        """Set column labels."""
        self._flush_pending()
        self._df.columns = new_columns

    def head(self, n: int = 5) -> "DataFrame":
        """Return the first n rows."""
        self._flush_pending()
        return DataFrame(self._df.head(n))

    def tail(self, n: int = 5) -> "DataFrame":
        """Return the last n rows."""
        self._flush_pending()
        return DataFrame(self._df.tail(n))

    @property
    def shape(self) -> tuple:
        """Return a tuple representing the dimensionality (rows, columns)."""
        n_pending_new = sum(1 for k in self._pending_cols if k not in self._df.columns)
        return (self._df.height, self._df.width + n_pending_new)

    def to_pandas(self):
        """Convert to pandas DataFrame."""
        self._flush_pending()
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
        self._flush_pending()
        return GroupBy(self._df.group_by(by))

    @property
    def loc(self):
        """Label-based indexing (pandas-like)."""
        self._flush_pending()
        return LocIndexer(self)

    @property
    def iloc(self):
        """Integer position-based indexing (pandas-like)."""
        self._flush_pending()
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
        self._flush_pending()
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
        self._flush_pending()
        keep_map = {"first": "first", "last": "last", False: "none", None: "first"}
        out = self._df.unique(subset=subset, keep=keep_map.get(keep, "first"), maintain_order=False)
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

    def nlargest(self, n: int, columns, keep: str = "first") -> "DataFrame":
        """Return the n rows with the largest values in the given column(s)."""
        cols = columns if isinstance(columns, list) else [columns]
        out = self._df.sort(cols, descending=True, nulls_last=True).head(n)
        return DataFrame(out)

    def nsmallest(self, n: int, columns, keep: str = "first") -> "DataFrame":
        """Return the n rows with the smallest values in the given column(s)."""
        cols = columns if isinstance(columns, list) else [columns]
        out = self._df.sort(cols, descending=False, nulls_last=True).head(n)
        return DataFrame(out)

    def sample(self, n: int | None = None, frac: float | None = None,
               replace: bool = False, random_state: int | None = None) -> "DataFrame":
        """Return a random sample of rows."""
        seed = random_state
        if frac is not None:
            out = self._df.sample(fraction=frac, with_replacement=replace, seed=seed)
        else:
            out = self._df.sample(n=n or 1, with_replacement=replace, seed=seed)
        return DataFrame(out)

    def pivot_table(self, values=None, index=None, columns=None,
                    aggfunc="mean", fill_value=None) -> "DataFrame":
        """
        Summarise data like pandas pivot_table.

        Supports string aggfunc: 'mean', 'sum', 'min', 'max', 'count'.
        When columns= is not given, returns a two-column DataFrame (index + value).
        When columns= is given, pivots into wide format.
        """
        aggfunc_map = {
            "mean":  pl.col(values).mean(),
            "sum":   pl.col(values).sum(),
            "min":   pl.col(values).min(),
            "max":   pl.col(values).max(),
            "count": pl.col(values).count(),
        }
        if isinstance(aggfunc, str):
            if aggfunc not in aggfunc_map:
                raise ValueError(f"aggfunc '{aggfunc}' not supported natively. Use: {list(aggfunc_map)}")
            agg_expr = aggfunc_map[aggfunc].alias(values)
        else:
            raise ValueError("aggfunc must be a string ('mean', 'sum', 'min', 'max', 'count')")

        group_cols = [index] if isinstance(index, str) else list(index)

        if columns is not None:
            # Wide format: group by index+columns, then pivot
            pivot_col = columns if isinstance(columns, str) else columns[0]
            grouped = self._df.group_by(group_cols + [pivot_col]).agg(agg_expr)
            out = grouped.pivot(on=pivot_col, index=group_cols, values=values)
            out = out.sort(group_cols)
        else:
            out = self._df.group_by(group_cols).agg(agg_expr).sort(group_cols)

        if fill_value is not None:
            out = out.fill_null(fill_value)

        return DataFrame(out)

    def describe(self, percentiles=(0.25, 0.5, 0.75)):
        """Summary statistics for numeric columns (native Polars)."""
        self._flush_pending()
        return DataFrame(self._df.describe(percentiles=list(percentiles)))

    def median(self):
        """Median of each numeric column (native Polars)."""
        self._flush_pending()
        numeric = [c for c in self._df.columns if self._df[c].dtype.is_numeric()]
        return DataFrame(self._df.select([pl.col(c).median() for c in numeric]))

    def std(self, ddof: int = 1):
        """Standard deviation of each numeric column (native Polars)."""
        self._flush_pending()
        numeric = [c for c in self._df.columns if self._df[c].dtype.is_numeric()]
        return DataFrame(self._df.select([pl.col(c).std(ddof=ddof) for c in numeric]))

    def corr(self) -> "DataFrame":
        """Pairwise Pearson correlation of numeric columns (native Polars).

        Returns a DataFrame with a leading 'feature' column (row labels) followed
        by one column per numeric input column — similar to pandas corr() but
        without a named index.
        """
        self._flush_pending()
        numeric = [c for c in self._df.columns if self._df[c].dtype.is_numeric()]
        rows = []
        for col_a in numeric:
            row = [col_a]
            for col_b in numeric:
                if col_a == col_b:
                    row.append(1.0)
                else:
                    row.append(self._df.select(pl.corr(col_a, col_b)).item())
            rows.append(row)
        data = {"feature": [r[0] for r in rows]}
        for i, col_b in enumerate(numeric):
            data[col_b] = [r[i + 1] for r in rows]
        return DataFrame(pl.DataFrame(data))

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
        self._flush_pending()
        right._flush_pending()
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
            # Optimize: use numpy values directly if available (faster than tolist())
            if hasattr(rows, 'values') and isinstance(rows.values, np.ndarray):
                mask = pl.Series("", rows.values).cast(pl.Boolean)
            else:
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
            # Optimize: use numpy values directly if available
            any_series = rows.any(axis=1)
            if hasattr(any_series, 'values') and isinstance(any_series.values, np.ndarray):
                mask = pl.Series("", any_series.values).cast(pl.Boolean)
            else:
                mask_values = any_series.tolist()
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

        # Process result and return appropriate type (Polars-first, avoid pandas when possible)
        if isinstance(result, pl.DataFrame):
            height, width = result.height, result.width

            # Single scalar
            if height == 1 and width == 1:
                return result.to_series(0)[0]

            # Single column -> return Polars Series (Pandas-like Series semantics with .to_list())
            if pl_cols is not None and isinstance(pl_cols, list) and len(pl_cols) == 1:
                series = result[pl_cols[0]]
                if series.len() == 1:
                    return series.to_list()[0]
                return series

            # General case: keep wrapped DataFrame (Polars backend)
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
        
        import polars as pl

        # Use Polars-native indexing for performance
        # Row selection
        df_rows = self.df._df[pl_rows]

        # Column selection
        df_result = df_rows.select(pl_cols)

        height, width = df_result.height, df_result.width

        # Single scalar
        if height == 1 and width == 1:
            return df_result.to_series(0)[0]

        # Single column -> return Polars Series
        if width == 1:
            series = df_result.to_series(0)
            if series.len() == 1:
                return series.to_list()[0]
            return series

        # General case: wrap in DataFrame
        return DataFrame(df_result)
