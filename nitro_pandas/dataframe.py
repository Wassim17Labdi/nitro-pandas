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

import builtins
import functools
import inspect
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

class PandasFallbackWarning(UserWarning):
    """Warning emitted when nitro-pandas falls back to pandas.

    Silence with:
        import warnings, nitro_pandas as npd
        warnings.filterwarnings("ignore", category=npd.PandasFallbackWarning)
    """


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class _NeedsPandas(Exception):
    """Raised by a native method when these arguments require the pandas implementation."""


def _warn_fallback(name):
    # The profiler matches warnings to the user's line, whatever the internal call depth.
    level, frame = 1, sys._getframe()
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    warnings.warn(
        f"[nitro-pandas] '{name}' is not natively implemented — pandas fallback activated.",
        PandasFallbackWarning,
        stacklevel=level,
    )


def _pandas_arg(value):
    if isinstance(value, (DataFrame, Series, pl.DataFrame, pl.Series)):
        return value.to_pandas()
    if type(value) in (list, tuple):
        return type(value)(_pandas_arg(v) for v in value)
    if type(value) is dict:
        return {k: _pandas_arg(v) for k, v in value.items()}
    return value


def _pandas_args(fn):
    """Run fn natively, or through pandas when called with arguments it does not declare."""
    params = list(inspect.signature(fn).parameters.values())[1:]
    accepted = {p.name for p in params}
    open_ended = any(p.kind is p.VAR_KEYWORD for p in params)
    max_positional = sum(p.kind is p.POSITIONAL_OR_KEYWORD for p in params)
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        unknown = set(kwargs) - accepted - {"inplace"}
        if (unknown and not open_ended) or len(args) > max_positional:
            return self._pandas_call(name, args, kwargs)
        native_kwargs = {k: v for k, v in kwargs.items() if k != "inplace"}
        self._flush_pending()
        try:
            result = fn(self, *args, **native_kwargs)
        except _NeedsPandas:
            return self._pandas_call(name, args, kwargs)
        if kwargs.get("inplace"):
            self._replace_with(result)
            return None
        return result

    return wrapper


def _from_pandas(pdf):
    if pdf.index.names != [None]:
        pdf = pdf.reset_index()
    pdf = pdf.reset_index(drop=True)
    try:
        return pl.from_pandas(pdf)
    except Exception:
        # Polars has no mixed-type column, unlike a pandas object column.
        objects = pdf.select_dtypes("object").columns
        return pl.from_pandas(pdf.astype({c: "string" for c in objects}))


_DTYPES = {
    "str": pl.String, "string": pl.String, "object": pl.String,
    "int": pl.Int64, "int8": pl.Int8, "int16": pl.Int16, "int32": pl.Int32, "int64": pl.Int64,
    "uint8": pl.UInt8, "uint16": pl.UInt16, "uint32": pl.UInt32, "uint64": pl.UInt64,
    "float": pl.Float64, "float32": pl.Float32, "float64": pl.Float64,
    "bool": pl.Boolean, "boolean": pl.Boolean,
    "datetime": pl.Datetime("us"), "datetime64": pl.Datetime("us"),
    "datetime64[ns]": pl.Datetime("ns"), "datetime64[us]": pl.Datetime("us"), "datetime64[ms]": pl.Datetime("ms"),
    "date": pl.Date, "category": pl.Categorical,
}
_PYTHON_DTYPES = {str: pl.String, int: pl.Int64, float: pl.Float64, bool: pl.Boolean}


def _polars_dtype(dtype):
    """Return the Polars type for a pandas dtype, or None when there is no safe mapping."""
    if isinstance(dtype, pl.DataType) or (isinstance(dtype, type) and issubclass(dtype, pl.DataType)):
        return dtype
    if dtype in _PYTHON_DTYPES:
        return _PYTHON_DTYPES[dtype]
    try:
        name = dtype.lower() if isinstance(dtype, str) else np.dtype(dtype).name
    except TypeError:
        return None
    return _DTYPES.get(name)


def _is_na(name, dtype):
    if dtype.is_float():
        return pl.col(name).is_null() | pl.col(name).is_nan()
    return pl.col(name).is_null()


def _is_numeric(dtype):
    return dtype.is_numeric() or dtype == pl.Boolean


def _fits(dtype, value):
    if isinstance(value, (bool, np.bool_)):
        return dtype == pl.Boolean
    if isinstance(value, (int, float, np.number)):
        return dtype.is_numeric()
    if isinstance(value, str):
        return dtype == pl.String
    return dtype.is_temporal()


def _fill_value(dtype, value):
    """Return the value to fill a column with; Polars has no mixed-type column."""
    if _fits(dtype, value):
        return value
    if dtype == pl.String and isinstance(value, (int, float, np.number)):
        return str(value)
    raise _NeedsPandas


def _column_expr(name, val):
    if isinstance(val, Series):
        return val._series.alias(name)
    if isinstance(val, (pl.Series, pl.Expr)):
        return val.alias(name)
    if isinstance(val, pd.Series):
        return pl.from_pandas(val).alias(name)
    if isinstance(val, (np.ndarray, list, tuple)):
        return pl.Series(name, val)
    return pl.lit(val).alias(name)


def _unwrap(other):
    if isinstance(other, Series):
        return other._series
    if isinstance(other, pd.Series):
        return pl.from_pandas(other)
    return other


class _StringAccessor:
    """Maps the pandas str API onto Polars; anything else runs through pandas."""

    def __init__(self, owner):
        self._owner = owner
        self._str = owner._series.str

    def _pandas(self, name, *args, **kwargs):
        _warn_fallback(f"str.{name}")
        return getattr(self._owner._series.to_pandas().str, name)(*_pandas_arg(list(args)), **_pandas_arg(kwargs))

    def contains(self, pat, case=True, flags=0, na=None, regex=True):
        if flags:
            return self._pandas("contains", pat, case=case, flags=flags, na=na, regex=regex)
        native = f"(?i){pat if regex else re.escape(pat)}" if not case else pat
        try:
            out = self._str.contains(native, literal=not regex and case)
        except pl.exceptions.ComputeError:
            # Rust regex has no lookaround or backreferences.
            return self._pandas("contains", pat, case=case, na=na, regex=regex)
        return Series(out if na is None else out.fill_null(na))

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=False):
        unsupported = flags or n not in (-1, 1) or not isinstance(pat, str) or not isinstance(repl, str)
        if unsupported or (regex and "\\" in repl):
            return self._pandas("replace", pat, repl, n=n, case=case, flags=flags, regex=regex)
        native, literal = pat, not regex
        if case is False:
            native, literal = f"(?i){pat if regex else re.escape(pat)}", False
        value = repl.replace("$", "$$") if not literal else repl
        method = self._str.replace_all if n == -1 else self._str.replace
        try:
            return Series(method(native, value, literal=literal))
        except pl.exceptions.ComputeError:
            return self._pandas("replace", pat, repl, n=n, case=case, flags=flags, regex=regex)

    def startswith(self, pat, na=None):
        if not isinstance(pat, str):
            return self._pandas("startswith", pat, na=na)
        out = self._str.starts_with(pat)
        return Series(out if na is None else out.fill_null(na))

    def endswith(self, pat, na=None):
        if not isinstance(pat, str):
            return self._pandas("endswith", pat, na=na)
        out = self._str.ends_with(pat)
        return Series(out if na is None else out.fill_null(na))

    def lower(self):  return Series(self._str.to_lowercase())
    def upper(self):  return Series(self._str.to_uppercase())
    def len(self):    return Series(self._str.len_chars())
    def zfill(self, width):  return Series(self._str.zfill(width))
    def strip(self, to_strip=None):   return Series(self._str.strip_chars(to_strip))
    def lstrip(self, to_strip=None):  return Series(self._str.strip_chars_start(to_strip))
    def rstrip(self, to_strip=None):  return Series(self._str.strip_chars_end(to_strip))

    def slice(self, start=None, stop=None, step=None):
        start_at = start or 0
        if step not in (None, 1) or start_at < 0 or (stop is not None and stop < 0):
            return self._pandas("slice", start, stop, step)
        return Series(self._str.slice(start_at, None if stop is None else builtins.max(stop - start_at, 0)))

    def split(self, pat=None, n=-1, expand=False, regex=None):
        if pat is None or len(pat) != 1 or n != -1 or expand or regex:
            return self._pandas("split", pat, n=n, expand=expand, regex=regex)
        return Series(self._str.split(pat))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.slice(key, key + 1) if key >= 0 else self._pandas("get", key)
        return self.slice(key.start, key.stop, key.step)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(pd.Series.str, name):
            def fallback(*args, **kwargs):
                return self._pandas(name, *args, **kwargs)
            return fallback
        return getattr(self._str, name)


_DATETIME_FIELDS = {
    "year": lambda dt: dt.year(), "month": lambda dt: dt.month(), "day": lambda dt: dt.day(),
    "hour": lambda dt: dt.hour(), "minute": lambda dt: dt.minute(), "second": lambda dt: dt.second(),
    "quarter": lambda dt: dt.quarter(), "dayofyear": lambda dt: dt.ordinal_day(),
    "dayofweek": lambda dt: dt.weekday() - 1, "weekday": lambda dt: dt.weekday() - 1,
}


class _DatetimeAccessor:
    """Exposes pandas datetime fields as properties, as pandas does (Polars uses methods)."""

    def __init__(self, owner):
        self._owner = owner
        self._dt = owner._series.dt

    def strftime(self, date_format):
        return Series(self._dt.to_string(date_format))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _DATETIME_FIELDS:
            return Series(_DATETIME_FIELDS[name](self._dt))
        if hasattr(pd.core.indexes.accessors.DatetimeProperties, name):
            _warn_fallback(f"dt.{name}")
            return getattr(self._owner._series.to_pandas().dt, name)
        return getattr(self._dt, name)


class Series:
    """Thin Polars-backed wrapper with a pandas-like API.

    Methods implemented here follow pandas semantics. Names known to pandas but
    not implemented here run through pandas; Polars-only names pass through.
    """

    def __init__(self, pl_series: pl.Series):
        self._series = pl_series

    def _flush_pending(self, read_col=None):
        pass

    def _replace_with(self, result):
        self._series = result._series

    def _pandas_call(self, name, args, kwargs):
        _warn_fallback(f"Series.{name}")
        pds = self._series.to_pandas()
        result = getattr(pds, name)(*_pandas_arg(list(args)), **_pandas_arg(kwargs))
        if kwargs.get("inplace"):
            self._series = pl.from_pandas(pds)
        return result

    def _valid(self):
        return self._series.fill_nan(None) if self._series.dtype.is_float() else self._series

    def _statistic(self, name, *args, **kwargs):
        if not _is_numeric(self._series.dtype) and name not in ("min", "max"):
            raise _NeedsPandas
        return getattr(self._valid(), name)(*args, **kwargs)

    @property
    def str(self):
        return _StringAccessor(self)

    @property
    def dt(self):
        return _DatetimeAccessor(self)

    @property
    def name(self):   return self._series.name
    @property
    def shape(self):  return (len(self._series),)
    @property
    def size(self):   return len(self._series)
    @property
    def empty(self):  return len(self._series) == 0
    @property
    def dtype(self):  return self._series.head(0).to_pandas().dtype
    @property
    def index(self):  return pd.RangeIndex(len(self._series))
    @property
    def values(self): return self._series.to_numpy()

    @_pandas_args
    def sum(self):     return self._statistic("sum")
    @_pandas_args
    def mean(self):    return self._statistic("mean")
    @_pandas_args
    def median(self):  return self._statistic("median")
    @_pandas_args
    def min(self):     return self._statistic("min")
    @_pandas_args
    def max(self):     return self._statistic("max")
    @_pandas_args
    def std(self, ddof=1):  return self._statistic("std", ddof=ddof)
    @_pandas_args
    def var(self, ddof=1):  return self._statistic("var", ddof=ddof)
    @_pandas_args
    def count(self):        return self._valid().drop_nulls().len()
    @_pandas_args
    def nunique(self):      return self._valid().drop_nulls().n_unique()
    @_pandas_args
    def unique(self):       return self._series.unique(maintain_order=True).to_numpy()

    @_pandas_args
    def quantile(self, q=0.5):
        if not isinstance(q, (int, float)):
            raise _NeedsPandas
        return self._statistic("quantile", q, interpolation="linear")

    @_pandas_args
    def abs(self):   return Series(self._series.abs())
    @_pandas_args
    def round(self, decimals=0):  return Series(self._series.round(decimals, mode="half_to_even"))
    @_pandas_args
    def clip(self, lower=None, upper=None):  return Series(self._series.clip(lower, upper))
    @_pandas_args
    def shift(self, periods=1):  return Series(self._series.shift(periods))
    @_pandas_args
    def diff(self, periods=1):   return Series(self._series.diff(n=periods))
    @_pandas_args
    def rank(self):    return Series(self._series.rank("average"))
    @_pandas_args
    def cumsum(self):  return Series(self._series.cum_sum())

    def isna(self):
        out = self._series.is_null()
        return Series(out | self._series.is_nan().fill_null(False) if self._series.dtype.is_float() else out)

    def notna(self):
        return ~self.isna()

    isnull = isna
    notnull = notna

    @_pandas_args
    def isin(self, values):
        items = values.to_list() if isinstance(values, (Series, pl.Series)) else list(values)
        try:
            return Series(self._series.is_in(items).fill_null(False))
        except (pl.exceptions.PolarsError, TypeError):
            raise _NeedsPandas

    @_pandas_args
    def between(self, left, right, inclusive="both"):
        return Series(self._series.is_between(left, right, closed=inclusive).fill_null(False))

    @_pandas_args
    def dropna(self):  return Series(self._valid().drop_nulls())

    @_pandas_args
    def fillna(self, value):
        fill = _fill_value(self._series.dtype, value)
        return Series(self._valid().fill_null(fill))

    @_pandas_args
    def sort_values(self, *, ascending=True, na_position="last"):
        return Series(self._series.sort(descending=not ascending, nulls_last=na_position == "last"))

    @_pandas_args
    def head(self, n=5):  return Series(self._series.head(n))
    @_pandas_args
    def tail(self, n=5):  return Series(self._series.tail(n))
    @_pandas_args
    def copy(self, deep=True):  return Series(self._series.clone())

    @_pandas_args
    def astype(self, dtype):
        target = _polars_dtype(dtype)
        if target is None:
            raise _NeedsPandas
        out = self._series.cast(target)
        if target == pl.String and self._series.dtype.is_numeric():
            out = out.fill_null("nan")
        return Series(out)

    @_pandas_args
    def value_counts(self, sort=True, ascending=False, dropna=True, normalize=False):
        source = self._valid().drop_nulls() if dropna else self._series
        out = source.value_counts(parallel=True)
        if normalize:
            out = out.with_columns((pl.col("count") / source.len()).alias("proportion")).drop("count")
        value = "proportion" if normalize else "count"
        if sort:
            out = out.sort(value, descending=not ascending)
        result = DataFrame(out)
        result._keyed = True
        return result

    def to_numpy(self):  return self._series.to_numpy()
    def __array__(self, dtype=None, copy=None):
        out = self._series.to_numpy()
        return out if dtype is None else out.astype(dtype)

    # A null compares as False in pandas (NaN), and as True for !=.
    def __gt__(self, other):  return Series((self._series > _unwrap(other)).fill_null(False))
    def __lt__(self, other):  return Series((self._series < _unwrap(other)).fill_null(False))
    def __ge__(self, other):  return Series((self._series >= _unwrap(other)).fill_null(False))
    def __le__(self, other):  return Series((self._series <= _unwrap(other)).fill_null(False))
    def __eq__(self, other):  return Series((self._series == _unwrap(other)).fill_null(False))
    def __ne__(self, other):  return Series((self._series != _unwrap(other)).fill_null(True))

    def __add__(self, other):       return Series(self._series + _unwrap(other))
    def __radd__(self, other):      return Series(_unwrap(other) + self._series)
    def __sub__(self, other):       return Series(self._series - _unwrap(other))
    def __rsub__(self, other):      return Series(_unwrap(other) - self._series)
    def __mul__(self, other):       return Series(self._series * _unwrap(other))
    def __rmul__(self, other):      return Series(_unwrap(other) * self._series)
    def __truediv__(self, other):   return Series(self._series / _unwrap(other))
    def __rtruediv__(self, other):  return Series(_unwrap(other) / self._series)
    def __floordiv__(self, other):  return Series(self._series // _unwrap(other))
    def __rfloordiv__(self, other): return Series(_unwrap(other) // self._series)
    def __mod__(self, other):       return Series(self._series % _unwrap(other))
    def __rmod__(self, other):      return Series(_unwrap(other) % self._series)
    def __pow__(self, other):       return Series(self._series ** _unwrap(other))
    def __rpow__(self, other):      return Series(_unwrap(other) ** self._series)
    def __neg__(self):              return Series(-self._series)
    def __pos__(self):              return self
    def __abs__(self):              return Series(self._series.abs())

    def __and__(self, other):   return Series(self._series & _unwrap(other))
    def __rand__(self, other):  return Series(_unwrap(other) & self._series)
    def __or__(self, other):    return Series(self._series | _unwrap(other))
    def __ror__(self, other):   return Series(_unwrap(other) | self._series)
    def __xor__(self, other):   return Series(self._series ^ _unwrap(other))
    def __invert__(self):       return Series(~self._series)

    def __getitem__(self, key):
        if isinstance(key, Series):
            return Series(self._series.filter(key._series))
        result = self._series[key]
        return Series(result) if isinstance(result, pl.Series) else result

    def __bool__(self):
        raise ValueError("The truth value of a Series is ambiguous. Use a.any() or a.all().")

    def __len__(self):   return len(self._series)
    def __iter__(self):  return iter(self._series)
    def __repr__(self):  return self._series.__repr__()
    def __str__(self):   return self._series.__str__()

    def to_pandas(self):  return self._series.to_pandas()
    def to_list(self):    return self._series.to_list()
    tolist = to_list

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if hasattr(pd.Series, name):
            if callable(getattr(pd.Series, name)):
                def fallback(*args, **kwargs):
                    return self._pandas_call(name, args, kwargs)
                return fallback
            _warn_fallback(f"Series.{name}")
            return getattr(self._series.to_pandas(), name)
        if hasattr(self._series, name):
            return getattr(self._series, name)
        raise AttributeError(f"'Series' object has no attribute '{name}'")


_AGGREGATIONS = {
    "mean": lambda c: c.mean(), "sum": lambda c: c.sum(), "min": lambda c: c.min(), "max": lambda c: c.max(),
    "count": lambda c: c.count(), "std": lambda c: c.std(), "var": lambda c: c.var(), "median": lambda c: c.median(),
    "first": lambda c: c.drop_nulls().first(), "last": lambda c: c.drop_nulls().last(),
    "nunique": lambda c: c.drop_nulls().n_unique(),
}
_NUMERIC_ONLY = ("std", "var", "median")


class _Grouped:
    """Shared grouping state: pandas sorts keys and drops null keys by default."""

    _pandas_class = pd.core.groupby.DataFrameGroupBy

    def __init__(self, source, by, sort, dropna, pandas_by=None):
        self._source = source
        self._by = by
        self._pandas_by = by if pandas_by is None else pandas_by
        self._sort = sort
        self._dropna = dropna
        frame = source.drop_nulls(subset=by) if dropna else source
        self._gb = frame.group_by(by, maintain_order=not sort)

    def _finish(self, out, value_col=None):
        if self._sort:
            out = out.sort(self._by, nulls_last=True)
        result = DataFrame(out)
        result._keyed = True
        result._value_col = value_col
        return result

    def _pandas_groupby(self):
        return self._source.to_pandas().groupby(self._pandas_by, sort=self._sort, dropna=self._dropna)

    def _pandas_call(self, name, args, kwargs):
        _warn_fallback(f"groupby.{name}")
        return getattr(self._pandas_groupby(), name)(*_pandas_arg(list(args)), **_pandas_arg(kwargs))

    def __iter__(self):
        _warn_fallback("groupby.__iter__")
        return iter(self._pandas_groupby())

    def __getattr__(self, name):
        if name.startswith("_") or not hasattr(self._pandas_class, name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if callable(getattr(self._pandas_class, name)):
            def fallback(*args, **kwargs):
                return self._pandas_call(name, args, kwargs)
            return fallback
        _warn_fallback(f"groupby.{name}")
        return getattr(self._pandas_groupby(), name)


class GroupBy(_Grouped):
    """pandas-style groupby on the Polars engine, e.g. df.groupby("a")["b"].mean()."""

    def __getitem__(self, key):
        if isinstance(key, (str, list)):
            return GroupByColumn(self, key)
        raise TypeError(f"GroupBy indices must be str or list, got {type(key)}")

    def _frame_aggregate(self, func, numeric_only=False):
        if func == "size":
            return self._finish(self._gb.agg(pl.len().alias("size")))
        columns = cs.numeric() if numeric_only or func in _NUMERIC_ONLY else pl.all()
        return self._finish(self._gb.agg(_AGGREGATIONS[func](columns)))

    def agg(self, *args, **kwargs):
        """Aggregate from a pandas dict, named aggregations, or Polars expressions."""
        native = _AGGREGATIONS
        if len(args) == 1 and isinstance(args[0], str) and not kwargs:
            return getattr(self, args[0])() if args[0] in native else self._pandas_call("agg", args, kwargs)
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            spec = args[0]
            if set(spec) & set(self._by) or not all(isinstance(f, str) and f in native for f in spec.values()):
                return self._pandas_call("agg", args, kwargs)
            return self._finish(self._gb.agg([native[f](pl.col(c)).alias(c) for c, f in spec.items()]))
        if not args and kwargs and all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values()):
            if not all(isinstance(f, str) and f in native for _, f in kwargs.values()):
                return self._pandas_call("agg", args, kwargs)
            return self._finish(self._gb.agg([native[f](pl.col(c)).alias(n) for n, (c, f) in kwargs.items()]))
        if all(isinstance(a, (pl.Expr, pl.Series)) for a in args) and (args or kwargs):
            return self._finish(self._gb.agg(*args, **kwargs))
        return self._pandas_call("agg", args, kwargs)

    aggregate = agg


class GroupByColumn(_Grouped):
    """Aggregations on selected columns: df.groupby("a")["b"].mean()."""

    _pandas_class = pd.core.groupby.SeriesGroupBy

    def __init__(self, parent, column):
        self._source = parent._source
        self._by = parent._by
        self._pandas_by = parent._pandas_by
        self._sort = parent._sort
        self._dropna = parent._dropna
        self._gb = parent._gb
        self._column = column

    def _pandas_groupby(self):
        return super()._pandas_groupby()[self._column]

    def _frame_aggregate(self, func, numeric_only=False):
        value_col = self._column if isinstance(self._column, str) else None
        if func == "size":
            return self._finish(self._gb.agg(pl.len().alias("size")), "size")
        columns = [self._column] if isinstance(self._column, str) else self._column
        exprs = [_AGGREGATIONS[func](pl.col(c)).alias(c) for c in columns]
        return self._finish(self._gb.agg(exprs), value_col)

    def agg(self, func=None, *args, **kwargs):
        if isinstance(func, str) and func in _AGGREGATIONS and not args and not kwargs:
            return self._frame_aggregate(func)
        return self._pandas_call("agg", (func, *args), kwargs)

    aggregate = agg


def _group_method(func):
    def method(self, *args, **kwargs):
        numeric_only = kwargs.get("numeric_only", False)
        if args or set(kwargs) - {"numeric_only"}:
            return self._pandas_call(func, args, kwargs)
        return self._frame_aggregate(func, numeric_only)
    method.__name__ = func
    return method


for _func in (*_AGGREGATIONS, "size"):
    setattr(GroupBy, _func, _group_method(_func))
    setattr(GroupByColumn, _func, _group_method(_func))


_PANDAS_MUTATORS = {"update"}


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
    
    def __init__(self, data=None, pl_df: pl.DataFrame = None, columns=None):
        """Build from a dict, list, array, Polars or pandas DataFrame."""
        if pl_df is not None:
            self._df = pl_df
        elif data is None:
            self._df = pl.DataFrame()
        elif isinstance(data, pl.DataFrame):
            self._df = data
        elif isinstance(data, pd.DataFrame):
            self._df = _from_pandas(data)
        else:
            self._df = pl.DataFrame(data, schema=columns, orient="row" if isinstance(data, list) and columns else None)
        # Buffer for pending column assignments — flushed as a single with_columns()
        self._pending_cols = {}
        self._keyed = False
        self._value_col = None

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
        exprs = [_column_expr(name, val) for name, val in self._pending_cols.items()]
        self._df = self._df.with_columns(exprs)
        self._pending_cols = {}

    def _replace_with(self, result):
        self._df = result._df
        self._pending_cols = {}

    def _pandas_call(self, name, args, kwargs):
        _warn_fallback(name)
        self._flush_pending()
        pdf = self._df.to_pandas()
        result = getattr(pdf, name)(*_pandas_arg(list(args)), **_pandas_arg(kwargs))
        if kwargs.get("inplace") or name in _PANDAS_MUTATORS:
            self._df = _from_pandas(pdf)
        return result

    def __getattr__(self, name: str):
        """Fall back to pandas for what is not implemented; columns resolve as attributes."""
        if name.startswith("_"):
            raise AttributeError(name)
        if not hasattr(pd.DataFrame, name):
            if name in self.columns:
                return self[name]
            raise AttributeError(f"'DataFrame' object has no attribute '{name}'")

        if callable(getattr(pd.DataFrame, name)):
            def _pandas_fallback(*args, **kwargs):
                return self._pandas_call(name, args, kwargs)
            return _pandas_fallback

        self._flush_pending()
        _warn_fallback(name)
        return getattr(self._df.to_pandas(), name)

    def __setattr__(self, name, value):
        if name.startswith("_") or hasattr(type(self), name) or name not in self.columns:
            object.__setattr__(self, name, value)
        else:
            self[name] = value

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
        self._flush_pending()
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

        self._flush_pending()
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
        self._flush_pending()
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
        self._flush_pending()
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
        self._flush_pending()
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
        self._flush_pending()
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
        self._flush_pending()
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
        self._flush_pending()
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
            self._flush_pending()
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
        renamed = self._df.clone()
        renamed.columns = list(new_columns)
        self._df = renamed

    @property
    def empty(self):
        return self.shape[0] == 0 or self.shape[1] == 0

    @property
    def size(self):
        return self.shape[0] * self.shape[1]

    @property
    def index(self):
        return pd.RangeIndex(self._df.height)

    @property
    def dtypes(self):
        self._flush_pending()
        return self._df.head(0).to_pandas().dtypes

    def __len__(self):
        return self._df.height

    def __iter__(self):
        return iter(self.columns)

    def __contains__(self, key):
        return key in self.columns

    def __delitem__(self, key):
        self._flush_pending()
        self._df = self._df.drop(key)

    def insert(self, loc, column, value, allow_duplicates=False):
        """Insert a column at position loc, in place."""
        self._flush_pending()
        if column in self._df.columns:
            raise ValueError(f"cannot insert {column}, already exists")
        order = self._df.columns
        order.insert(loc, column)
        self._df = self._df.with_columns(_column_expr(column, value)).select(order)

    def pop(self, item):
        """Remove a column and return it."""
        self._flush_pending()
        column = Series(self._df[item])
        self._df = self._df.drop(item)
        return column

    def copy(self, deep=True):
        """Return a copy (Polars frames are immutable, so this costs nothing)."""
        self._flush_pending()
        return DataFrame(self._df.clone())

    def assign(self, **kwargs):
        """Return a copy with new columns; callables receive the frame being built."""
        out = self.copy()
        for name, value in kwargs.items():
            out[name] = value(out) if callable(value) else value
        out._flush_pending()
        return out

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

    @_pandas_args
    def to_csv(self, path=None, sep=None, na_rep=None, header=True, index=None, **kwargs):
        """Write a CSV without index column; index=True goes through pandas."""
        pandas_only = {"columns", "encoding", "float_format", "mode", "compression", "quoting", "quotechar",
                       "date_format", "lineterminator", "decimal", "chunksize", "escapechar", "doublequote",
                       "errors", "storage_options", "index_label"}
        if index or pandas_only & kwargs.keys() or not isinstance(header, bool):
            raise _NeedsPandas
        if sep is not None:
            kwargs.setdefault("separator", sep)
        if na_rep is not None:
            kwargs.setdefault("null_value", na_rep)
        kwargs.setdefault("include_header", header)
        return self._df.write_csv(path, **kwargs)

    @_pandas_args
    def to_parquet(self, path, **kwargs):
        """Write DataFrame to a Parquet file using Polars backend."""
        if "partition_cols" in kwargs:
            raise _NeedsPandas
        kwargs.pop("engine", None)
        kwargs.pop("index", None)
        self._df.write_parquet(path, **kwargs)

    @_pandas_args
    def to_json(self, path, **kwargs):
        """Write DataFrame to a JSON file using Polars backend."""
        if {"orient", "lines", "date_format", "indent", "compression"} & kwargs.keys():
            raise _NeedsPandas
        self._df.write_json(path, **kwargs)

    @_pandas_args
    def to_excel(self, path, **kwargs):
        """Write DataFrame to an Excel file (Polars has no native writer used here)."""
        kwargs.setdefault("index", False)
        self._df.to_pandas().to_excel(path, **kwargs)

    @_pandas_args
    def groupby(self, by=None, sort=True, dropna=True, as_index=True, observed=None, group_keys=True):
        """Group by column name(s). Keys stay as columns, there is no index to move them to."""
        keys = [by] if isinstance(by, str) else by
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise _NeedsPandas
        return GroupBy(self._df, keys, sort, dropna, pandas_by=by)

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

    @_pandas_args
    def sort_values(self, by=None, *, ascending=True, na_position="last", ignore_index=False):
        """Sort by one or more columns."""
        if by is None:
            if self._value_col is None:
                raise TypeError("sort_values() missing 1 required positional argument: 'by'")
            by = self._value_col
        cols = by if isinstance(by, list) else [by]
        descending = [not a for a in ascending] if isinstance(ascending, list) else not ascending
        return DataFrame(self._df.sort(by=cols, descending=descending, nulls_last=na_position == "last"))

    @_pandas_args
    def rename(self, mapper=None, columns=None, axis=None):
        """Rename columns; a mapper without axis targets the index, which does not exist here."""
        if mapper is not None and axis in (1, "columns"):
            columns = mapper
        if not columns:
            return DataFrame(self._df)
        return DataFrame(self._df.rename(columns, strict=False))

    @_pandas_args
    def drop(self, labels=None, axis=0, index=None, columns=None, errors="raise"):
        """Drop columns, or rows by position."""
        if columns is not None or axis in (1, "columns"):
            target = columns if columns is not None else labels
            names = [target] if isinstance(target, str) else list(target)
            return DataFrame(self._df.drop(names, strict=errors == "raise"))

        rows = index if index is not None else labels
        if rows is None:
            return DataFrame(self._df)
        if axis not in (0, "index"):
            raise ValueError(f"axis must be 0 (rows) or 1 (columns), got {axis}")
        if isinstance(rows, slice):
            rows = list(range(*rows.indices(self._df.height)))
        if isinstance(rows, int):
            rows = [rows]
        if not isinstance(rows, list):
            raise ValueError(f"Unsupported row labels: {type(rows)}")
        return DataFrame(self._df.filter(~pl.int_range(0, self._df.height).is_in(rows)))

    @_pandas_args
    def astype(self, dtype):
        """Cast columns with pandas or Polars type names."""
        spec = dtype if isinstance(dtype, dict) else {c: dtype for c in self._df.columns}
        exprs = []
        for name, wanted in spec.items():
            target = _polars_dtype(wanted)
            if target is None:
                raise _NeedsPandas
            expr = pl.col(name).cast(target)
            if target == pl.String and self._df.schema[name].is_numeric():
                expr = expr.fill_null("nan")
            exprs.append(expr)
        return DataFrame(self._df.with_columns(exprs))

    @_pandas_args
    def fillna(self, value):
        """Fill missing values with a scalar, or a per-column dict or Series."""
        if isinstance(value, (pd.Series, Series)):
            value = dict(zip(value.index, value.tolist()))
        spec = value if isinstance(value, dict) else {c: value for c in self._df.columns}
        schema = self._df.schema
        exprs = []
        for name, fill in spec.items():
            if not self._df[name].null_count() and not schema[name].is_float():
                continue
            source = pl.col(name).fill_nan(None) if schema[name].is_float() else pl.col(name)
            exprs.append(source.fill_null(_fill_value(schema[name], fill)).alias(name))
        return DataFrame(self._df.with_columns(exprs))

    @_pandas_args
    def dropna(self, *, how="any", subset=None):
        """Drop rows with missing values."""
        names = self._df.columns if subset is None else [subset] if isinstance(subset, str) else list(subset)
        schema = self._df.schema
        missing = [_is_na(c, schema[c]) for c in names]
        keep = ~pl.any_horizontal(missing) if how == "any" else ~pl.all_horizontal(missing)
        return DataFrame(self._df.filter(keep))

    @_pandas_args
    def drop_duplicates(self, subset=None, keep="first", ignore_index=False):
        """Remove duplicate rows, keeping the original row order as pandas does."""
        keep_map = {"first": "first", "last": "last", False: "none"}
        names = [subset] if isinstance(subset, str) else subset
        return DataFrame(self._df.unique(subset=names, keep=keep_map[keep], maintain_order=True))

    @_pandas_args
    def duplicated(self, subset=None, keep="first"):
        """Return a boolean Series marking duplicate rows."""
        if keep not in ("first", "last"):
            raise _NeedsPandas
        names = self._df.columns if subset is None else [subset] if isinstance(subset, str) else list(subset)
        row = pl.struct(names)
        distinct = row.is_first_distinct() if keep == "first" else row.is_last_distinct()
        return Series(self._df.select((~distinct).alias("duplicated")).to_series())

    @_pandas_args
    def value_counts(self, subset=None, sort=True, ascending=False, dropna=True, normalize=False, column=None):
        """Count rows per combination of values."""
        target = column if column is not None else subset
        names = self._df.columns if target is None else [target] if isinstance(target, str) else list(target)
        frame = self._df.drop_nulls(subset=names) if dropna else self._df
        out = frame.group_by(names).agg(pl.len().alias("count"))
        if normalize:
            out = out.with_columns((pl.col("count") / frame.height).alias("proportion")).drop("count")
        if sort:
            out = out.sort("proportion" if normalize else "count", descending=not ascending)
        result = DataFrame(out)
        result._keyed = True
        return result

    @_pandas_args
    def nlargest(self, n, columns, keep="first") -> "DataFrame":
        """Return the n rows with the largest values in the given column(s)."""
        cols = columns if isinstance(columns, list) else [columns]
        return DataFrame(self._df.sort(cols, descending=True, nulls_last=True).head(n))

    @_pandas_args
    def nsmallest(self, n, columns, keep="first") -> "DataFrame":
        """Return the n rows with the smallest values in the given column(s)."""
        cols = columns if isinstance(columns, list) else [columns]
        return DataFrame(self._df.sort(cols, descending=False, nulls_last=True).head(n))

    @_pandas_args
    def sample(self, n=None, frac=None, replace=False, random_state=None) -> "DataFrame":
        """Return a random sample of rows."""
        if frac is not None:
            return DataFrame(self._df.sample(fraction=frac, with_replacement=replace, seed=random_state))
        return DataFrame(self._df.sample(n=n or 1, with_replacement=replace, seed=random_state))

    @_pandas_args
    def pivot_table(self, values=None, index=None, columns=None, aggfunc="mean", fill_value=None) -> "DataFrame":
        """Summarise data like pandas pivot_table, with index keys kept as columns."""
        if not isinstance(values, str) or index is None or aggfunc not in _AGGREGATIONS:
            raise _NeedsPandas
        agg_expr = _AGGREGATIONS[aggfunc](pl.col(values)).alias(values)
        group_cols = [index] if isinstance(index, str) else list(index)
        pivot_col = None if columns is None else columns if isinstance(columns, str) else columns[0]
        keys = group_cols if pivot_col is None else group_cols + [pivot_col]
        grouped = self._df.drop_nulls(subset=keys).group_by(keys).agg(agg_expr)
        if pivot_col is None:
            out = grouped.sort(group_cols)
        else:
            ordered = grouped[pivot_col].unique().sort().cast(pl.String).to_list()
            out = grouped.pivot(on=pivot_col, index=group_cols, values=values)
            populated = [c for c in ordered if out[c].null_count() < out.height]
            out = out.select(group_cols + populated).sort(group_cols)
        if fill_value is not None:
            out = out.fill_null(fill_value)
        result = DataFrame(out)
        result._keyed = True
        return result

    def _numeric_columns(self, numeric_only):
        schema = self._df.schema
        numeric = [c for c in self._df.columns if _is_numeric(schema[c])]
        if not numeric_only and len(numeric) != self._df.width:
            raise _NeedsPandas
        return numeric

    def _statistic(self, func, numeric_only, *args, **kwargs):
        numeric = self._numeric_columns(numeric_only)
        if not numeric:
            return pd.Series(dtype=float)
        schema = self._df.schema
        exprs = [
            getattr(pl.col(c).fill_nan(None) if schema[c].is_float() else pl.col(c), func)(*args, **kwargs)
            for c in numeric
        ]
        return pd.Series(self._df.select(exprs).row(0), index=numeric)

    @_pandas_args
    def sum(self, numeric_only=False):     return self._statistic("sum", numeric_only)
    @_pandas_args
    def mean(self, numeric_only=False):    return self._statistic("mean", numeric_only)
    @_pandas_args
    def min(self, numeric_only=False):     return self._statistic("min", numeric_only)
    @_pandas_args
    def max(self, numeric_only=False):     return self._statistic("max", numeric_only)
    @_pandas_args
    def median(self, numeric_only=False):  return self._statistic("median", numeric_only)
    @_pandas_args
    def std(self, ddof=1, numeric_only=False):  return self._statistic("std", numeric_only, ddof=ddof)
    @_pandas_args
    def var(self, ddof=1, numeric_only=False):  return self._statistic("var", numeric_only, ddof=ddof)

    @_pandas_args
    def count(self):
        """Count non-missing values per column."""
        schema = self._df.schema
        counts = self._df.select([(~_is_na(c, schema[c])).sum().alias(c) for c in self._df.columns]).row(0)
        return pd.Series(counts, index=self._df.columns)

    @_pandas_args
    def nunique(self):
        """Count distinct non-null values per column."""
        counts = self._df.select([pl.col(c).drop_nulls().n_unique() for c in self._df.columns]).row(0)
        return pd.Series(counts, index=self._df.columns)

    @_pandas_args
    def describe(self, percentiles=(0.25, 0.5, 0.75)):
        """Summary statistics of numeric columns, in the layout pandas returns."""
        schema = self._df.schema
        numeric = [c for c in self._df.columns if schema[c].is_numeric()]
        if not numeric or any(schema[c].is_temporal() for c in self._df.columns):
            raise _NeedsPandas
        points = sorted({*percentiles, 0.5})
        stats = self._df.select(numeric).describe(percentiles=points, interpolation="linear")
        summary = stats.filter(pl.col("statistic") != "null_count").to_pandas().set_index("statistic")
        summary.index.name = None
        return summary.astype(float)

    @_pandas_args
    def corr(self, numeric_only=False):
        """Pairwise Pearson correlation, dropping missing values pair by pair as pandas does."""
        numeric = self._numeric_columns(numeric_only)
        matrix = np.eye(len(numeric))
        for i, left in enumerate(numeric):
            for j in range(i + 1, len(numeric)):
                right = numeric[j]
                pair = self._df.select(pl.col(left).cast(pl.Float64), pl.col(right).cast(pl.Float64)).drop_nulls()
                matrix[i, j] = matrix[j, i] = pair.select(pl.corr(left, right)).item()
        return pd.DataFrame(matrix, index=numeric, columns=numeric)

    @_pandas_args
    def reset_index(self, drop=False, name="index", names=None):
        """Add row numbers as a column; grouped results already carry their keys as columns."""
        if drop or self._keyed:
            return DataFrame(self._df)
        label = names if isinstance(names, str) else name
        return DataFrame(self._df.with_row_index(label).with_columns(pl.col(label).cast(pl.Int64)))

    @_pandas_args
    def merge(self, right, how="inner", on=None, left_on=None, right_on=None, suffixes=("_x", "_y")):
        """Join with pandas semantics for suffixes, key columns, null keys and row order."""
        if isinstance(right, pd.DataFrame):
            right = DataFrame(_from_pandas(right))
        right._flush_pending()
        left_frame, right_frame = self._df, right._df
        if how == "cross":
            shared = set(left_frame.columns) & set(right_frame.columns)
            left_frame = left_frame.rename({c: f"{c}{suffixes[0]}" for c in shared})
            right_frame = right_frame.rename({c: f"{c}{suffixes[1]}" for c in shared})
            return DataFrame(left_frame.join(right_frame, how="cross"))
        if on is None and left_on is None:
            on = [c for c in left_frame.columns if c in right_frame.columns]
        left_keys = [on] if isinstance(on, str) else on if on is not None else [left_on] if isinstance(left_on, str) else list(left_on)
        right_keys = left_keys if on is not None else [right_on] if isinstance(right_on, str) else list(right_on)

        same_named = {l for l, r in zip(left_keys, right_keys) if l == r}
        if len(same_named) not in (0, len(left_keys)):
            raise _NeedsPandas
        shared = (set(left_frame.columns) & set(right_frame.columns)) - same_named
        left_frame = left_frame.rename({c: f"{c}{suffixes[0]}" for c in shared})
        right_frame = right_frame.rename({c: f"{c}{suffixes[1]}" for c in shared})
        left_names = [f"{k}{suffixes[0]}" if k in shared else k for k in left_keys]
        right_names = [f"{k}{suffixes[1]}" if k in shared else k for k in right_keys]

        order = {"inner": "left", "left": "left", "right": "right"}.get(how, "none")
        out = left_frame.join(
            right_frame, how="full" if how == "outer" else how, left_on=left_names, right_on=right_names,
            coalesce=bool(same_named), nulls_equal=True, maintain_order=order,
        )
        if how == "outer":
            out = out.sort(left_names, nulls_last=True)
        return DataFrame(out)

    @staticmethod
    def concat(dfs: list["DataFrame"], axis: int = 0):
        """Concatenate DataFrames (kept for compatibility, prefer npd.concat())."""
        from . import concat as module_concat
        return module_concat(dfs, axis=axis)

    def _null_mask(self, invert):
        self._flush_pending()
        schema = self._df.schema
        exprs = [(~_is_na(c, schema[c]) if invert else _is_na(c, schema[c])).alias(c) for c in self._df.columns]
        return DataFrame(self._df.select(exprs))

    def isna(self):
        """Return boolean DataFrame indicating missing values."""
        return self._null_mask(invert=False)

    def notna(self):
        """Return boolean DataFrame indicating non-missing values."""
        return self._null_mask(invert=True)

    isnull = isna
    notnull = notna


def _row_mask(frame, rows):
    """Turn a loc row selector into a boolean Polars Series over the whole frame."""
    height = frame._df.height
    if isinstance(rows, Series):
        rows = rows._series
    if isinstance(rows, pd.Series):
        rows = pl.Series(rows.to_numpy())
    if isinstance(rows, pl.Series) and rows.dtype == pl.Boolean:
        return rows.fill_null(False)
    if isinstance(rows, np.ndarray) and rows.dtype == bool:
        return pl.Series(rows)
    if isinstance(rows, list) and rows and isinstance(rows[0], (bool, np.bool_)):
        return pl.Series(rows, dtype=pl.Boolean)
    positions = pl.int_range(0, height, eager=True)
    if isinstance(rows, slice):
        start = rows.start if rows.start is not None else 0
        stop = rows.stop if rows.stop is not None else height - 1
        return positions.is_between(start, stop)
    if isinstance(rows, (int, np.integer)):
        return positions == int(rows)
    if isinstance(rows, list):
        return positions.is_in(rows)
    raise NotImplementedError(f"loc: unsupported row selection type: {type(rows)}")


def _assign_rows(frame, mask, names, value):
    value = _unwrap(value)
    selected = int(mask.sum())
    exprs = []
    for name in names:
        exists = name in frame._df.columns
        if isinstance(value, (pl.Series, np.ndarray, list, tuple)) and len(value) == selected != frame._df.height:
            values = pl.Series(name, value)
            current = frame._df[name] if exists else pl.Series(name, [None] * frame._df.height, dtype=values.dtype)
            exprs.append(current.cast(pl.Float64 if values.dtype.is_float() and current.dtype.is_numeric() else current.dtype)
                         .scatter(mask.arg_true(), values).alias(name))
            continue
        replacement = pl.lit(value) if isinstance(value, pl.Series) else _column_expr(name, value)
        current = pl.col(name) if exists else pl.lit(None)
        exprs.append(pl.when(pl.lit(mask)).then(replacement).otherwise(current).alias(name))
    frame._df = frame._df.with_columns(exprs)


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
        if isinstance(rows, Series):
            rows = rows._series
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

        elif isinstance(rows, slice):
            if rows == slice(None):
                selected = self.df._df
            elif rows.step in (None, 1):
                start = rows.start if rows.start is not None else 0
                stop = rows.stop if rows.stop is not None else self.df._df.height - 1
                selected = self.df._df.slice(start, builtins.max(stop + 1 - start, 0))
            else:
                selected = self.df._df[list(range(rows.start or 0, (self.df._df.height if rows.stop is None else rows.stop + 1), rows.step))]
            result = selected if pl_cols is None else selected.select(pl_cols)

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

        if isinstance(rows, (int, np.integer)):
            if isinstance(cols, str):
                return result.item()
            return pd.Series(result.row(0), index=result.columns)
        if isinstance(cols, str):
            return Series(result[cols])
        return DataFrame(result)

    def __setitem__(self, key, value):
        rows, cols = key if isinstance(key, tuple) else (key, None)
        names = self.df._df.columns if cols is None else [cols] if isinstance(cols, str) else list(cols)
        _assign_rows(self.df, _row_mask(self.df, rows), names, value)


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
        
        if isinstance(rows, slice) and rows.step in (None, 1):
            start, stop, _ = rows.indices(self.df._df.height)
            df_rows = self.df._df.slice(start, builtins.max(stop - start, 0))
        else:
            df_rows = self.df._df[pl_rows]
        df_result = df_rows.select(pl_cols)

        if isinstance(rows, int):
            if isinstance(cols, int):
                return df_result.item()
            return pd.Series(df_result.row(0), index=df_result.columns)
        if isinstance(cols, int):
            return Series(df_result.to_series(0))
        return DataFrame(df_result)

    def __setitem__(self, key, value):
        rows, cols = key if isinstance(key, tuple) else (key, slice(None))
        columns = self.df._df.columns
        if isinstance(cols, int):
            names = [columns[cols]]
        elif isinstance(cols, slice):
            names = columns[cols]
        else:
            names = [columns[i] for i in cols]
        positions = pl.int_range(0, self.df._df.height, eager=True)
        if isinstance(rows, slice):
            selected = list(range(*rows.indices(self.df._df.height)))
        else:
            selected = [rows] if isinstance(rows, int) else rows
        _assign_rows(self.df, positions.is_in(selected), names, value)
