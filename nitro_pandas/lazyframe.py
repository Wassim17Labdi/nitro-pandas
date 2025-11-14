"""
LazyFrame module for nitro-pandas.

This module provides a pandas-like LazyFrame wrapper around Polars LazyFrames.
LazyFrames enable lazy evaluation, where operations are not executed until
collect() is called, allowing for query optimization.

Key features:
- Pandas-like API for lazy operations
- Polars LazyFrame backend for query optimization
- Support for lazy groupby operations
- Query optimization through lazy evaluation
"""

from .dataframe import DataFrame
import polars as pl
import re


class LazyGroupByColumn:
    """
    Wrapper for column-specific lazy groupby operations.
    
    Enables pandas-style syntax: lf.groupby("col1")["col2"].mean()
    All operations use Polars LazyFrame backend for optimization.
    """
    
    def __init__(self, gb, column):
        """
        Initialize LazyGroupByColumn wrapper.
        
        Args:
            gb: Polars LazyGroupBy object
            column: Column name (str) or list of column names
        """
        import polars as pl
        self._gb = gb
        self._column = column
    
    def mean(self):
        """Compute mean for selected column(s) in each group (lazy)."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).mean())
            return LazyFrame(result)
        else:
            result = self._gb.agg([pl.col(col).mean().alias(col) for col in self._column])
            return LazyFrame(result)
    
    def sum(self):
        """Compute sum for selected column(s) in each group (lazy)."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).sum())
            return LazyFrame(result)
        else:
            result = self._gb.agg([pl.col(col).sum().alias(col) for col in self._column])
            return LazyFrame(result)
    
    def min(self):
        """Compute minimum for selected column(s) in each group (lazy)."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).min())
            return LazyFrame(result)
        else:
            result = self._gb.agg([pl.col(col).min().alias(col) for col in self._column])
            return LazyFrame(result)
    
    def max(self):
        """Compute maximum for selected column(s) in each group (lazy)."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).max())
            return LazyFrame(result)
        else:
            result = self._gb.agg([pl.col(col).max().alias(col) for col in self._column])
            return LazyFrame(result)
    
    def count(self):
        """Count rows for selected column(s) in each group (lazy)."""
        import polars as pl
        if isinstance(self._column, str):
            result = self._gb.agg(pl.col(self._column).count().alias(self._column))
            return LazyFrame(result)
        else:
            result = self._gb.agg([pl.col(col).count().alias(col) for col in self._column])
            return LazyFrame(result)


class LazyGroupBy:
    """
    Wrapper for Polars LazyGroupBy operations with pandas-like API.
    
    This class enables pandas-style lazy groupby operations like:
    - lf.groupby("col").mean()
    - lf.groupby("col")["other_col"].mean()
    - lf.groupby("col").agg({"col": "mean"})
    
    All operations use Polars LazyFrame backend for query optimization.
    """
    
    def __init__(self, gb):
        """
        Initialize LazyGroupBy wrapper.
        
        Args:
            gb: Polars LazyGroupBy object
        """
        self._gb = gb
    
    def __getitem__(self, key):
        """
        Support for pandas-like column selection: lf.groupby("col1")["col2"]
        
        Args:
            key: Column name (str) or list of column names
            
        Returns:
            LazyGroupByColumn: Wrapper for column-specific lazy groupby operations
            
        Raises:
            TypeError: If key is not str or list
        """
        import polars as pl
        if isinstance(key, str):
            return LazyGroupByColumn(self._gb, key)
        elif isinstance(key, list):
            return LazyGroupByColumn(self._gb, key)
        else:
            raise TypeError(f"LazyGroupBy indices must be str or list, got {type(key)}")
    
    def mean(self):
        """Compute mean for all numeric columns in each group (lazy)."""
        return LazyFrame(self._gb.mean())
    
    def sum(self):
        """Compute sum for all numeric columns in each group (lazy)."""
        return LazyFrame(self._gb.sum())
    
    def min(self):
        """Compute minimum for all numeric columns in each group (lazy)."""
        return LazyFrame(self._gb.min())
    
    def max(self):
        """Compute maximum for all numeric columns in each group (lazy)."""
        return LazyFrame(self._gb.max())
    
    def count(self):
        """Count rows in each group (lazy)."""
        return LazyFrame(self._gb.count())
    
    def agg(self, *args, **kwargs):
        """
        Aggregate operations with pandas-like dictionary syntax (lazy).
        
        Supports both pandas-style dict aggregation and Polars expressions:
        - lf.groupby("col").agg({"col": "mean"})  # pandas-style
        - lf.groupby("col").agg(pl.col("col").mean())  # Polars-style
        
        Args:
            *args: Aggregation expressions or dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            LazyFrame: Lazy aggregated results
            
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
                    # Apply callable function directly
                    # Note: Polars requires return_dtype for UDFs (User Defined Functions)
                    # Default to Float64 for numeric operations (e.g., lambda x: x.sum() / len(x))
                    pl_expressions.append(pl.col(col).map_elements(func, return_dtype=pl.Float64))
                else:
                    raise ValueError(f"Unsupported function type: {type(func)}")
            return LazyFrame(self._gb.agg(pl_expressions))
        
        # Pass through Polars expressions directly
        return LazyFrame(self._gb.agg(*args, **kwargs))


class LazyFrame:
    """
    Pandas-like LazyFrame wrapper around Polars LazyFrame.
    
    This class provides a pandas-compatible API for lazy operations while
    using Polars LazyFrame as the backend. Lazy evaluation allows Polars
    to optimize queries before execution.
    
    Key features:
    - Pandas-like API for lazy operations
    - Polars LazyFrame backend for query optimization
    - Lazy evaluation until collect() is called
    - Support for lazy groupby and filtering operations
    
    Example:
        >>> import nitro_pandas as npd
        >>> lf = npd.read_csv_lazy("data.csv")
        >>> result = lf.filter(lf["col"] > 2).groupby("cat").mean()
        >>> df = result.collect()  # Execute query
    """
    
    def __init__(self, pl_lf: pl.LazyFrame):
        """
        Initialize LazyFrame wrapper.
        
        Args:
            pl_lf: Polars LazyFrame object
        """
        self._lf = pl_lf

    def __getitem__(self, key):
        """
        Boolean filtering with Polars expressions.
        
        Supports filtering with Polars expressions:
        - lf[lf["col"] > 2]
        - lf[pl.col("col") > 2]
        
        Args:
            key: Polars expression for filtering
            
        Returns:
            LazyFrame: Filtered LazyFrame
        """
        if isinstance(key, pl.Expr):
            return LazyFrame(self._lf.filter(key))
        return self._lf[key]

    def __repr__(self):
        """
        String representation of LazyFrame (pandas-like display).
        
        Returns a readable string representation of the LazyFrame,
        showing the query plan and schema information.
        
        Returns:
            str: String representation of LazyFrame
        """
        # Use Polars' built-in string representation which shows the query plan
        return self._lf.__repr__()
    
    def __str__(self):
        """
        String representation of LazyFrame.
        
        Returns:
            str: String representation of LazyFrame
        """
        return self._lf.__str__()

    def groupby(self, by):
        """
        Group LazyFrame by one or more columns.
        
        Args:
            by: Column name(s) to group by
            
        Returns:
            LazyGroupBy: LazyGroupBy object for aggregation operations
        """
        return LazyGroupBy(self._lf.group_by(by))

    def collect(self) -> DataFrame:
        """
        Execute lazy operations and return DataFrame.
        
        This method triggers the execution of all lazy operations
        and returns a DataFrame with the results.
        
        Returns:
            DataFrame: Executed results as DataFrame
        """
        return DataFrame(self._lf.collect())

    def query(self, expr: str):
        """
        Filter LazyFrame using a query expression (pandas-like).
        
        Converts pandas-style query strings to Polars expressions.
        Supports 'and'/'or' operators and column references.
        
        Args:
            expr: Query string (e.g., "col1 > 2 and col2 == 'A'")
            
        Returns:
            LazyFrame: Filtered LazyFrame
            
        Example:
            >>> lf.query("id > 2 and name == 'Bob'")
        """
        import re
        
        # Replace column names with pl.col() expressions
        def repl(match):
            col = match.group(0)
            return f'pl.col("{col}")'
        
        # Build regex pattern to match column names
        columns = self._lf.schema.keys()
        pattern = r'\b(' + '|'.join(map(re.escape, columns)) + r')\b'
        expr_polars = re.sub(pattern, repl, expr)
        
        # Convert pandas-style operators to Polars syntax
        expr_polars = re.sub(r'\s+and\s+', ' & ', expr_polars)
        expr_polars = re.sub(r'\s+or\s+', ' | ', expr_polars)
        expr_polars = re.sub(r'\band\b', '&', expr_polars)
        expr_polars = re.sub(r'\bor\b', '|', expr_polars)
        
        # Add parentheses around comparison expressions
        expr_polars = re.sub(
            r'(pl\.col\([^)]+\)\s*[<>=!]=?\s*[^&|()]+)',
            lambda m: '(' + m.group(1).strip() + ')',
            expr_polars
        )
        
        # Evaluate expression and filter
        import polars as pl
        mask = eval(expr_polars, {"pl": pl})
        return LazyFrame(self._lf.filter(mask))

    def __getattr__(self, name):
        """
        Automatic fallback to Polars LazyFrame for unimplemented methods.
        
        This enables access to all Polars LazyFrame methods not explicitly
        implemented in nitro-pandas. Results are wrapped in LazyFrame if
        they are LazyFrames.
        
        Args:
            name: Method or attribute name
            
        Returns:
            Method or attribute from Polars LazyFrame
        """
        attr = getattr(self._lf, name)
        if callable(attr):
            def method(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pl.LazyFrame):
                    return LazyFrame(result)
                return result
            return method
        return attr
