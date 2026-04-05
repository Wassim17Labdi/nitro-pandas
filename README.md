
<div align="center">

![nitro-pandas Logo](https://raw.githubusercontent.com/Wassim17Labdi/nitro-pandas/main/docs/nitro-pandas-logo.png)

**A high-performance pandas-like DataFrame library powered by Polars**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Combine the familiar pandas API with Polars' blazing-fast performance*

</div>

---

## ✨ Features

- 🐼 **Pandas-like API** - Use familiar pandas syntax without learning a new library
- ⚡ **Polars Backend** - Leverage Polars' optimized engine for maximum performance
- 🔄 **Lazy Evaluation** - Optimize queries with lazy operations before execution
- 📊 **Comprehensive I/O** - Read/write CSV, Parquet, JSON, and Excel files
- 🎯 **Automatic Fallback** - Seamless fallback to pandas for unimplemented methods
- 🔧 **Type Safety** - Support for pandas-like type casting and schema inference

## 🎯 Why nitro-pandas?

**nitro-pandas** bridges the gap between pandas' user-friendly API and Polars' exceptional performance. If you're familiar with pandas but need better performance, nitro-pandas is the perfect solution.

### Performance Comparison

Benchmarked on the [Books Rating dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) (~3M rows, 10 columns). All times are wall-clock seconds on a single machine.

#### Core Operations

| Operation | nitro-pandas | pandas | Polars | vs pandas | vs Polars |
|-----------|-------------|--------|--------|-----------|-----------|
| Read CSV | 4.56s | 13.54s | 1.09s | **3.0x faster** | 0.24x |
| GroupBy + Count | 0.038s | 0.150s | 0.036s | **3.9x faster** | ~same |
| Chained Ops (filter+groupby+sort) | 0.049s | 0.089s | 0.014s | **1.8x faster** | 0.29x |
| GroupBy Multi-Column | 0.156s | 0.224s | 0.074s | **1.4x faster** | 0.48x |
| Sort Values | 0.082s | 0.178s | 0.082s | **2.2x faster** | ~same |
| Double Filter + GroupBy | 0.021s | 0.061s | 0.011s | **2.9x faster** | 0.53x |
| Value Counts | 0.010s | 0.007s | 0.010s | ~same | ~same |
| Multi Aggregations (mean/min/max) | 0.114s | 0.170s | 0.039s | **1.5x faster** | 0.35x |
| Nunique (count distinct) | 0.098s | 0.503s | 0.080s | **5.1x faster** | 0.81x |
| Drop Duplicates | 0.189s | 0.531s | 0.223s | **2.8x faster** | **1.2x faster** |
| Column Arithmetic | 0.010s | 0.002s | 0.003s | 0.19x | 0.28x |
| Fill Null Values | 0.011s | 0.005s | 0.003s | 0.42x | 0.29x |
| String Contains Filter | 0.635s | 0.574s | 0.022s | ~same | 0.03x |
| Describe (summary stats) | 0.088s | 0.074s | 0.014s | ~same | 0.16x |
| Select + Rename Columns | 0.001s | 0.035s | 0.001s | **47.8x faster** | ~same |
| **TOTAL** | **6.06s** | **16.14s** | **1.70s** | **2.7x faster** | 0.28x |

#### Extended Operations (native implementations)

| Operation | nitro-pandas | pandas | Polars | vs pandas | vs Polars |
|-----------|-------------|--------|--------|-----------|-----------|
| nlargest (top-N rows) | 0.059s | 0.141s | 0.179s | **2.4x faster** | **3.1x faster** |
| sample (random sampling) | 0.035s | 0.048s | 0.033s | **1.4x faster** | ~same |
| pivot_table (group aggregation) | 0.009s | 0.028s | 0.007s | **3.0x faster** | ~same |

#### Fallback Operations (via pandas)

| Operation | nitro-pandas | pandas | vs pandas |
|-----------|-------------|--------|-----------|
| median | 0.042s | 0.034s | ~same |
| std | 0.034s | 0.028s | ~same |
| corr | 0.020s | 0.015s | ~same |
| apply | 0.024s | 0.019s | ~same |
| cumsum | 0.023s | 0.014s | ~same |

> **Summary:** nitro-pandas is **faster than pandas in 10/15 core tests** with an overall **2.7x speedup** on the total benchmark. Operations implemented natively (groupby, sort, filter, nunique, nlargest, pivot_table) see the biggest gains. Fallback operations (median, std, corr, apply, cumsum) carry minimal overhead (~20%) over raw pandas.

*Results may vary based on data size and hardware.*

## 📦 Installation

```bash
# Using uv (recommended)
uv add nitro-pandas

# Using pip
pip install nitro-pandas
```

### Requirements

- **Python 3.11+**
- **Dependencies** (automatically installed):
  - `polars>=1.30.0` - High-performance DataFrame engine
  - `pandas>=2.2.3` - For fallback methods
  - `fastexcel>=0.7.0` - Fast Excel reading
  - `openpyxl>=3.1.5` - Excel file support
  - `pyarrow>=20.0.0` - Parquet file support

## 🚀 Quick Start

### Basic Usage

```python
import nitro_pandas as npd

# Create a DataFrame (pandas-like syntax)
df = npd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Paris', 'London', 'New York']
})

# Access columns (returns pandas Series for compatibility)
ages = df['age']
print(ages > 30)  # Boolean Series

# Filter data
filtered = df.loc[df['age'] > 30]
print(filtered)
```

### Reading Files

```python
# Read CSV
df = npd.read_csv('data.csv')

# Read with lazy evaluation (optimized for large files)
lf = npd.read_csv_lazy('large_data.csv')
df = lf.query('id > 1000').collect()

# Read other formats
df_parquet = npd.read_parquet('data.parquet')
df_excel = npd.read_excel('data.xlsx')
df_json = npd.read_json('data.json')
```

### Data Operations

```python
# GroupBy operations (pandas-like syntax, Polars backend)
result = df.groupby('city')['age'].mean()
print(result)

# Multi-column groupby
result = df.groupby(['city', 'category'])['value'].sum()

# Aggregations with dictionaries
result = df.groupby('category').agg({
    'value': 'mean',
    'count': 'sum'
})

# Sorting and filtering
df_sorted = df.sort_values('age', ascending=False)
df_filtered = df.query("age > 25 and city == 'Paris'")
```

### Writing Files

```python
# Write to various formats
df.to_csv('output.csv')
df.to_parquet('output.parquet')
df.to_json('output.json')
df.to_excel('output.xlsx')
```

## 📚 API Reference

### DataFrame Operations

#### Creation
```python
# From dictionary
df = npd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# From Polars DataFrame
df = npd.DataFrame(pl.DataFrame({'a': [1, 2, 3]}))

# Empty DataFrame
df = npd.DataFrame()
```

#### Indexing
```python
# Column selection
df['column_name']  # Returns pandas Series
df[['col1', 'col2']]  # Returns DataFrame

# Boolean filtering
df[df['age'] > 30]  # Returns DataFrame

# Label-based indexing
df.loc[df['age'] > 30, 'name']  # Returns Series
df.loc[0:5, ['name', 'age']]  # Returns DataFrame

# Position-based indexing
df.iloc[0:5, 0:2]  # Returns DataFrame
```

#### Transformations
```python
# Type casting (pandas-like types)
df = df.astype({'id': 'int64', 'name': 'str'})

# Rename columns
df = df.rename(columns={'old_name': 'new_name'})

# Drop rows/columns
df = df.drop(labels=[0, 1], axis=0)  # Drop rows
df = df.drop(labels=['col1'], axis=1)  # Drop columns

# Fill null values
df = df.fillna({'column': 0})

# Sort values
df = df.sort_values('age', ascending=False)
```

### I/O Functions

#### CSV
```python
# Eager reading
df = npd.read_csv('file.csv', 
                  sep=',',
                  usecols=['col1', 'col2'],
                  dtype={'id': 'int64'})

# Lazy reading
lf = npd.read_csv_lazy('file.csv', n_rows=1000)
df = lf.collect()
```

#### Parquet
```python
# Eager reading
df = npd.read_parquet('file.parquet',
                      columns=['col1', 'col2'],
                      n_rows=1000)

# Lazy reading
lf = npd.read_parquet_lazy('file.parquet')
df = lf.collect()
```

#### Excel
```python
# Eager reading
df = npd.read_excel('file.xlsx',
                    sheet_name=0,
                    usecols=['col1', 'col2'],
                    nrows=1000)

# Lazy reading
lf = npd.read_excel_lazy('file.xlsx', sheet_name='Sheet1')
df = lf.collect()
```

#### JSON
```python
# Eager reading
df = npd.read_json('file.json',
                   dtype={'id': 'int64'},
                   n_rows=1000)

# Lazy reading
lf = npd.read_json_lazy('file.json', lines=True)
df = lf.collect()
```

### LazyFrame Operations

```python
# Create lazy frame
lf = npd.read_csv_lazy('large_file.csv')

# Chain operations (optimized before execution)
result = (lf
          .query('age > 30')
          .groupby('city')
          .agg({'value': 'mean'}))

# Execute query
df = result.collect()
# Sort after collection if needed
df = df.sort_values('value', ascending=False)
```

## 🔄 Migration from pandas

Migrating from pandas to nitro-pandas is straightforward:

```python
# Before (pandas)
import pandas as pd
df = pd.read_csv('data.csv')
result = df.groupby('category')['value'].mean()

# After (nitro-pandas)
import nitro_pandas as npd
df = npd.read_csv('data.csv')
result = df.groupby('category')['value'].mean()
```

Most pandas operations work the same way! The main differences:

- **Single column selection** (`df['col']`) returns a pandas Series (not a nitro-pandas Series) to maintain compatibility with pandas expressions and boolean indexing
- **Comparison operations** (`df > 2`) return pandas DataFrames for boolean indexing compatibility
- **Unimplemented methods**: Automatic fallback to pandas is available at **both the DataFrame instance level and the package level**:
  ```python
  # ✅ Works: fallback on DataFrame instance
  df = npd.DataFrame({'a': [1, 2, 3]})
  result = df.describe()  # Falls back to pandas DataFrame method
  
  # ✅ Works: fallback at package level
  import pandas as pd
  df_pd = pd.DataFrame({'a': [1, 2, 1], 'b': ['x', 'y', 'x']})
  result = npd.get_dummies(df_pd)  # Falls back to pandas module function
  result = npd.date_range('2024-01-01', periods=5)  # Falls back to pandas
  ```
  Note: Methods that only exist on DataFrame instances (like `describe()`) are only available via DataFrame instances, not at the package level.
- **Mixed types in columns**: Unlike pandas, Polars (and thus nitro-pandas) does **not** allow mixed types within a single column. Each column must have a consistent type. If your pandas DataFrame has mixed types in a column, Polars will coerce them to a common type (usually `object`/string) or raise an error.
  ```python
  # ❌ This works in pandas but NOT in Polars/nitro-pandas
  pd.DataFrame({'col': [1, 'text', 3.5]})  # Mixed int, str, float
  
  # ✅ Polars will coerce to string or raise error
  npd.DataFrame({'col': [1, 'text', 3.5]})  # All values become strings
  ```
- **No `inplace` parameter**: Polars operations are always immutable (return new DataFrames), so nitro-pandas does **not** support the `inplace=True` parameter found in pandas. All operations return new DataFrame objects.
  ```python
  # ❌ This works in pandas but NOT in nitro-pandas
  df.drop(columns=['col'], inplace=True)  # inplace not supported
  
  # ✅ Always assign the result
  df = df.drop(labels=['col'], axis=1)  # Returns new DataFrame
  ```

## 🏗️ Project Structure

```
nitro-pandas/
├── nitro_pandas/
│   ├── __init__.py          # Package initialization
│   ├── dataframe.py         # DataFrame implementation
│   ├── lazyframe.py         # LazyFrame implementation
│   └── io/
│       ├── __init__.py      # IO module exports
│       ├── csv.py           # CSV I/O
│       ├── parquet.py       # Parquet I/O
│       ├── json.py          # JSON I/O
│       └── excel.py         # Excel I/O
├── tests/
│   ├── test_dataframe.py    # DataFrame tests
│   ├── test_groupby.py      # GroupBy tests
│   ├── test_io.py           # I/O tests
│   └── helpers.py           # Test utilities
├── pyproject.toml           # Project configuration
└── README.md                 # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/nitro-pandas.git
cd nitro-pandas

# Install development dependencies
uv sync --dev

# Run tests
uv run python tests/test_runner.py
```

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive open-source license that allows anyone to:
- ✅ Use the software for any purpose (commercial or personal)
- ✅ Modify the software
- ✅ Distribute the software
- ✅ Sublicense the software

**In short: Everyone can use it freely!**

## 🙏 Acknowledgments

- [Polars](https://www.pola.rs/) - For the high-performance DataFrame engine
- [pandas](https://pandas.pydata.org/) - For the API inspiration and fallback support

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for the Python data science community**

⭐ Star this repo if you find it useful!

</div>
