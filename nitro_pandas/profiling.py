"""
nitro-pandas profiling: line-by-line performance comparison between pandas and nitro-pandas.

Usage:
    import nitro_pandas as npd

    def workload(pd):
        df = pd.read_csv("data.csv")
        result = df.groupby("Id")["Price"].mean()
        return result

    result = npd.profile_compare(workload)
    print(result)

Requires: pip install 'nitro-pandas[profile]'
"""

import inspect
import warnings
from dataclasses import dataclass


@dataclass
class ProfileRow:
    lineno: int
    source: str
    pandas_s: float
    nitro_s: float
    gain: float       # pandas_s / nitro_s  →  >1 means nitro is faster
    fell_back: bool


@dataclass
class ProfileComparison:
    rows: list
    pandas_total: float
    nitro_total: float
    gain_total: float

    def __str__(self):
        header = f"{'Line':>5}  {'Source':<46}  {'pandas':>9}  {'nitro':>9}  {'Gain':>7}  {''}"
        sep = "-" * (len(header) + 2)
        lines = [sep, header, sep]
        for row in self.rows:
            src = row.source[:45]
            gain_str = f"{row.gain:.2f}x" if row.gain != float("inf") else "   inf"
            arrow = "↑" if row.gain >= 1.05 else ("↓" if row.gain < 0.95 else "~")
            fb = "⚠" if row.fell_back else " "
            lines.append(
                f"{row.lineno:>5}  {src:<46}  {row.pandas_s:>8.4f}s"
                f"  {row.nitro_s:>8.4f}s  {gain_str:>7}  {arrow}{fb}"
            )
        lines.append(sep)
        gain_str = f"{self.gain_total:.2f}x" if self.gain_total != float("inf") else "   inf"
        lines.append(
            f"{'TOTAL':>5}  {'':46}  {self.pandas_total:>8.4f}s"
            f"  {self.nitro_total:>8.4f}s  {gain_str:>7}"
        )
        lines.append(sep)
        note = (
            "\nNote: per-line timing is indicative — chained calls (.groupby().mean()) split\n"
            "      the cost across lines. The TOTAL is the reliable headline figure."
        )
        return "\n".join(lines) + note

    def to_dataframe(self):
        import nitro_pandas as npd
        return npd.DataFrame({
            "lineno":    [r.lineno for r in self.rows],
            "source":    [r.source for r in self.rows],
            "pandas_s":  [r.pandas_s for r in self.rows],
            "nitro_s":   [r.nitro_s for r in self.rows],
            "gain":      [r.gain for r in self.rows],
            "fell_back": [r.fell_back for r in self.rows],
        })


def _run_profiled(workload, module, n_runs, extra_functions):
    """Wrap workload in LineProfiler, run n_runs times, return (result, stats)."""
    try:
        from line_profiler import LineProfiler
    except ImportError:
        raise ImportError(
            "line_profiler is required for profile_compare. "
            "Install it with: pip install 'nitro-pandas[profile]'"
        )

    lp = LineProfiler()
    if extra_functions:
        for fn in extra_functions:
            lp.add_function(fn)
    lp_wrapper = lp(workload)  # wraps and registers workload

    result = None
    for _ in range(n_runs):
        result = lp_wrapper(module)

    return result, lp.get_stats()


def _extract_timings(stats, workload, n_runs):
    """Return {absolute_lineno: seconds} for the workload function."""
    unit = stats.unit
    _, start_lineno = inspect.getsourcelines(workload)
    func_name = workload.__name__

    timings_key = None
    for key in stats.timings:
        # key = (filename, first_lineno, func_name)
        if key[1] == start_lineno and key[2] == func_name:
            timings_key = key
            break

    if timings_key is None:
        return {}

    return {
        lineno: (ticks * unit) / n_runs
        for lineno, _nhits, ticks in stats.timings[timings_key]
    }


def _assert_results_equal(pd_result, npd_result):
    """Raise AssertionError if the two results differ meaningfully.

    Compares sorted numeric values to handle format differences between backends
    (e.g. pandas groupby returns a Series while nitro-pandas returns a DataFrame).
    Index differences are ignored — only values are checked.
    """
    import numpy as np
    import pandas as pd

    # Convert nitro result to pandas for comparison
    if hasattr(npd_result, "to_pandas"):
        npd_as_pd = npd_result.to_pandas()
    else:
        npd_as_pd = npd_result

    def _sorted_numeric_values(obj):
        if isinstance(obj, pd.DataFrame):
            vals = obj.select_dtypes(include="number").values.flatten()
        elif isinstance(obj, pd.Series):
            vals = obj.values.astype(float)
        else:
            return np.array([float(obj)])
        return np.sort(vals[~np.isnan(vals)])

    try:
        pd_vals = _sorted_numeric_values(pd_result)
        npd_vals = _sorted_numeric_values(npd_as_pd)
        if len(pd_vals) != len(npd_vals) or not np.allclose(pd_vals, npd_vals, rtol=1e-5):
            raise AssertionError(
                f"Results differ between pandas and nitro-pandas backends.\n"
                f"pandas:\n{pd_result}\n\nnitro-pandas:\n{npd_as_pd}"
            )
    except (TypeError, ValueError):
        # Non-numeric: fall back to shape check
        if hasattr(pd_result, "shape") and hasattr(npd_as_pd, "shape"):
            if pd_result.shape != npd_as_pd.shape:
                raise AssertionError(
                    f"Result shapes differ: pandas={pd_result.shape}, nitro={npd_as_pd.shape}"
                )


def profile_compare(
    workload,
    *,
    n_runs=1,
    warmup=0,
    assert_equal=False,
    extra_functions=None,
    return_format="table",
    min_time_s=1e-4,
):
    """
    Profile workload line-by-line under pandas and nitro-pandas, reporting per-line speedup.

    Args:
        workload: Callable that accepts a pandas-compatible module as its first argument.
                  Example: def workload(pd): return pd.read_csv("data.csv")
        n_runs: Number of measured runs (times are averaged). Default 1.
        warmup: Number of warm-up runs before measuring (discarded). Default 0.
        assert_equal: If True, raise AssertionError when both backends return different
                      values. Useful to catch regressions. Default False.
        extra_functions: Additional functions to profile alongside workload. Default None.
        return_format: "table" (default) — returns ProfileComparison (printable table).
                       "dict"  — returns list of dicts.
                       "dataframe" — returns npd.DataFrame.

    Returns:
        ProfileComparison | list[dict] | npd.DataFrame

    Requires:
        pip install 'nitro-pandas[profile]'
    """
    import pandas as _pd
    import nitro_pandas as _npd
    from nitro_pandas.dataframe import PandasFallbackWarning

    # Warm-up (results discarded, profiler not active)
    for _ in range(warmup):
        workload(_pd)
        workload(_npd)

    # Pandas run
    pd_result, pd_stats = _run_profiled(workload, _pd, n_runs, extra_functions)

    # Nitro run — capture fallback warnings and their line numbers
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PandasFallbackWarning)
        npd_result, npd_stats = _run_profiled(workload, _npd, n_runs, extra_functions)

    # Line numbers where a fallback was triggered (stacklevel=2 points to caller)
    fallback_linenos = {
        w.lineno for w in caught if issubclass(w.category, PandasFallbackWarning)
    }

    if assert_equal:
        _assert_results_equal(pd_result, npd_result)

    # Extract per-line timings
    pd_timings = _extract_timings(pd_stats, workload, n_runs)
    npd_timings = _extract_timings(npd_stats, workload, n_runs)

    # Source lines of the workload
    src_lines, start_lineno = inspect.getsourcelines(workload)

    rows = []
    for lineno in sorted(set(pd_timings) | set(npd_timings)):
        pd_s = pd_timings.get(lineno, 0.0)
        npd_s = npd_timings.get(lineno, 0.0)
        # Skip continuation lines of multi-line expressions (dict entries, closing
        # brackets, etc.) — they carry no meaningful timing signal.
        if pd_s < min_time_s and npd_s < min_time_s:
            continue
        offset = lineno - start_lineno
        source = src_lines[offset].strip() if 0 <= offset < len(src_lines) else ""
        gain = pd_s / npd_s if npd_s > 0 else float("inf")
        rows.append(ProfileRow(
            lineno=lineno,
            source=source,
            pandas_s=pd_s,
            nitro_s=npd_s,
            gain=gain,
            fell_back=lineno in fallback_linenos,
        ))

    pd_total = sum(r.pandas_s for r in rows)
    npd_total = sum(r.nitro_s for r in rows)
    gain_total = pd_total / npd_total if npd_total > 0 else float("inf")

    comparison = ProfileComparison(
        rows=rows,
        pandas_total=pd_total,
        nitro_total=npd_total,
        gain_total=gain_total,
    )

    if return_format == "dict":
        return [
            {
                "lineno": r.lineno,
                "source": r.source,
                "pandas_s": r.pandas_s,
                "nitro_s": r.nitro_s,
                "gain": r.gain,
                "fell_back": r.fell_back,
            }
            for r in rows
        ]
    if return_format == "dataframe":
        return comparison.to_dataframe()

    return comparison
