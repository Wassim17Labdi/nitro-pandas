# Changelog

All notable changes to nitro-pandas are documented here.

## [1.0.0] — 2026-11-14

Compatibility release: pandas code runs unchanged. Tested against pandas on 297 everyday calls: 0 crashes (58 in 0.2.x), same results as pandas.

### Added
- GroupBy: `size`, `nunique`, `first`, `last`, `std`, `var`, `median`, named aggregations, `as_index`, `sort`, `dropna`, iteration, and a pandas fallback for everything else
- pandas arguments on native methods: `inplace=True`, `drop(columns=...)`, `to_csv(index=False, sep=...)`, `numeric_only`; unknown arguments run through pandas instead of failing
- `len(df)`, `df.col`, `for col in df`, `"col" in df`, `del df[col]`, `insert`, `pop`, `copy`, `assign`, `duplicated`, `dropna`, `mean`, `sum`, `min`, `max`, `count`, `nunique`
- `df.loc[mask, col] = value` and `df.iloc[i, j] = value`
- `.str`: `lower`, `upper`, `strip`, `startswith`, `endswith`, `len`, slicing, `replace`; `.dt` fields as pandas properties (`.dt.year`, `.dt.dayofweek`...)
- `read_csv`: `parse_dates`, pandas missing-value markers (`NA`, `null`...), dtypes applied at parse time (keeps leading zeros); pandas-only options run through pandas
- `npd.merge`, `npd.concat(ignore_index=...)`; pandas module functions (`cut`, `get_dummies`...) accept nitro-pandas objects
- `tests/test_pandas_parity.py`: compares results with pandas

### Changed
- GroupBy results are sorted by key and drop null keys, as in pandas
- `drop_duplicates` keeps row order; comparisons treat missing values like pandas (`~mask` keeps them)
- `merge` applies pandas suffixes, matches null keys and keeps pandas row order
- `reset_index()` adds an `index` column unless `drop=True` (grouped results keep their keys)
- `rename(mapping)` without `columns=` no longer renames columns, as in pandas
- `mean`, `sum`, `median`, `std`, `count`, `nunique`, `describe` and `corr` return pandas objects, as in pandas

### Fixed
- Pending column assignments were ignored by `rename`, `drop`, `astype`, `fillna` and other methods
- `str.replace` replaced only the first match; `Series.head()` returned 10 rows
- Fallback warnings now point at the user's line, so `profile_compare` flags them reliably

## [0.2.1] — 2026-11-09

### Fixed
- Fresh installs picked up Polars 2 and pandas 3, which break nitro-pandas: dependencies pinned to `polars>=1.30.0,<1.36` and `pandas>=2.2.3,<3`

## [0.2.0] — 2026-05-20

### Added
- `npd.profile_compare()` — line-by-line performance comparison between pandas and nitro-pandas using `line_profiler` (optional dependency: `pip install 'nitro-pandas[profile]'`)
- Native `std()`, `median()`, `corr()`, `describe()` — no longer fall back to pandas
- `Series` wrapper class — column operations stay in Polars instead of returning `pd.Series`
- `_StringAccessor` — `str.contains(case=False)` translated to Polars regex natively

### Changed
- `df["col"] = ...` assignments are now batched and flushed lazily — reduces `with_columns()` calls on multi-column pipelines
- `drop_duplicates()` uses `maintain_order=False` for faster deduplication

## [0.1.6] — 2026-04-05

### Added
- Native `nlargest()`, `nsmallest()`, `sample()`, `pivot_table()` with full test coverage and benchmarks

### Fixed
- `query()` now uses AST parsing instead of `eval()` — blocks query injection attacks

## [0.1.5] — 2026-01-27

### Changed
- Optimized `loc` / `iloc` indexing
- Faster filtering with numpy boolean arrays

## [0.1.4] — 2025-11-14

### Added
- CI/CD workflows via GitHub Actions

## [0.1.3] — 2025-11-14

### Added
- `PandasFallbackWarning` — emitted when an operation falls back to pandas
- Optimized pandas fallback path

## [0.1.0] — 2025-11-14

### Added
- pandas-compatible `DataFrame` API backed by Polars
- `groupby`, `merge`, `sort_values`, `drop_duplicates`, `rename`, `fillna`, `query`
- `read_csv`, `read_excel`, `read_parquet`
- `GroupBy` with `mean`, `sum`, `count`, `min`, `max`, `agg`
