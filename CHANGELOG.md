# Changelog

All notable changes to nitro-pandas are documented here.

## [Unreleased]

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
