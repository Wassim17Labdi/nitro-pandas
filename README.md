
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

- 🐼 **Pandas-like API** — Use familiar pandas syntax without learning a new library
- ⚡ **Polars Backend** — Leverage Polars' optimized Rust engine for maximum performance
- 📊 **Comprehensive I/O** — Read/write CSV, Parquet, JSON, and Excel files
- 🎯 **Automatic Fallback** — Seamless fallback to pandas for unimplemented methods
- 🔬 **Built-in Profiler** — Line-by-line comparison of your pandas code vs nitro-pandas with `profile_compare`
- 🔧 **Type Safety** — Support for pandas-like type casting and schema inference

## 🎯 Why nitro-pandas?

**nitro-pandas** bridges the gap between pandas' user-friendly API and Polars' exceptional performance. Replace `import pandas as pd` with `import nitro_pandas as npd` and get faster code without changing anything else.

### Performance Comparison

Benchmarked on the [Books Rating dataset](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) (~3M rows, 10 columns) using `npd.profile_compare`. All times are averaged wall-clock seconds.

| Operation | pandas | nitro-pandas | Speedup |
|-----------|--------|-------------|---------|
| Read CSV | 13.04s | 1.09s | **12.0x** ↑ |
| Rename columns | 0.11s | 0.001s | **80x** ↑ |
| Drop duplicates | 0.52s | 0.42s | **1.2x** ↑ |
| Filter (`df[df["Price"] > 0]`) | 0.028s | 0.009s | **3.1x** ↑ |
| Chained filters | 0.008–0.014s | 0.001–0.002s | **5–8x** ↑ |
| GroupBy + mean | 0.030s | 0.004s | **7.5x** ↑ |
| GroupBy + count | 0.032s | 0.004s | **8.9x** ↑ |
| nlargest (top-N) | 0.027–0.029s | 0.007–0.009s | **3–4x** ↑ |
| String filter (`str.contains`) | 0.15–0.20s | 0.02–0.03s | **5–9x** ↑ |
| pivot_table | 0.008s | 0.003s | **2.8x** ↑ |
| sample (50k rows) | 0.010s | 0.001s | **8.0x** ↑ |
| describe | 0.012s | 0.003s | **4.2x** ↑ |
| sort_values | 0.041s | 0.018s | **2.3x** ↑ |
| **TOTAL pipeline** | **14.47s** | **1.68s** | **8.6x** ↑ |

> **Summary:** **8.6x overall speedup** on a realistic 13-step production pipeline. The biggest gains are on I/O, string operations, groupby, and sampling — the operations that dominate real-world workloads.

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
  - `polars>=1.30.0` — High-performance DataFrame engine
  - `pandas>=2.2.3` — For fallback methods
  - `line-profiler>=5.0.2` — For `profile_compare`
  - `fastexcel>=0.7.0` — Fast Excel reading
  - `openpyxl>=3.1.5` — Excel file support
  - `pyarrow>=20.0.0` — Parquet file support

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

# Filter data
filtered = df[df['age'] > 30]

# GroupBy
result = df.groupby('city')['age'].mean()
```

### Reading Files

```python
# Read CSV
df = npd.read_csv('data.csv')

# Read with lazy evaluation (optimized for large files)
lf = npd.read_csv_lazy('large_data.csv')
df = lf.query('id > 1000').collect()

# Other formats
df_parquet = npd.read_parquet('data.parquet')
df_excel   = npd.read_excel('data.xlsx')
df_json    = npd.read_json('data.json')
```

### Data Operations

```python
# GroupBy operations
result = df.groupby('city')['age'].mean()
result = df.groupby(['city', 'category'])['value'].sum()
result = df.groupby('category').agg({'value': 'mean', 'count': 'sum'})

# Sorting, filtering, sampling
df_sorted   = df.sort_values('age', ascending=False)
df_filtered = df.query("age > 25 and city == 'Paris'")
df_sample   = df.sample(n=1000, random_state=42)

# Top-N rows
top10 = df.nlargest(10, 'age')

# Pivot table
pivot = df.pivot_table(values='age', index='city', aggfunc='mean')

# Summary statistics
df.describe()
df.std()
df.median()
df.corr()
```

### Writing Files

```python
df.to_csv('output.csv')
df.to_parquet('output.parquet')
df.to_json('output.json')
df.to_excel('output.xlsx')
```

## 🔬 profile_compare

`profile_compare` runs your pandas code line-by-line under both backends and reports the speedup per line — so you know exactly where nitro-pandas helps.

```python
import nitro_pandas as npd

def my_pipeline(pd):
    df = pd.read_csv("data.csv")
    df = df.rename(columns={"review/score": "score"})
    result = df.groupby("Id")["score"].mean()
    return result

print(npd.profile_compare(my_pipeline))
```

Output:
```
------------------------------------------------------------------------------------------
 Line  Source                                               pandas      nitro     Gain  
------------------------------------------------------------------------------------------
    5  df = pd.read_csv("data.csv")                       13.0385s    1.0866s   12.00x  ↑ 
    6  df = df.rename(columns={"review/score": "sco       0.1051s    0.0013s   80.14x  ↑ 
    7  result = df.groupby("Id")["score"].mean()           0.0296s    0.0039s    7.51x  ↑ 
------------------------------------------------------------------------------------------
TOTAL                                                      13.1732s    1.0918s   12.07x
------------------------------------------------------------------------------------------
```

**Options:**

```python
npd.profile_compare(
    my_pipeline,
    n_runs=3,           # average over 3 runs
    warmup=1,           # 1 warm-up run discarded
    assert_equal=True,  # raise if results differ between backends
    return_format="dataframe",  # "table" (default) | "dict" | "dataframe"
)
```

Lines marked `⚠` triggered a pandas fallback — those are candidates for native implementation.

## 📚 API Reference

### DataFrame Operations

#### Creation
```python
df = npd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df = npd.DataFrame(pl.DataFrame({'a': [1, 2, 3]}))
```

#### Indexing
```python
df['column_name']           # Returns nitro-pandas Series
df[['col1', 'col2']]        # Returns DataFrame
df[df['age'] > 30]          # Boolean filtering
df.loc[df['age'] > 30, 'name']
df.iloc[0:5, 0:2]
```

#### Transformations
```python
df.astype({'id': 'int64', 'name': 'str'})
df.rename(columns={'old': 'new'})
df.drop(labels=['col1'], axis=1)
df.fillna({'column': 0})
df.sort_values('age', ascending=False)
df.drop_duplicates(subset=['id'])
```

#### Aggregations (native, no pandas fallback)
```python
df.describe()
df.std()
df.median()
df.corr()
df.groupby('col')['val'].mean()
df.groupby('col')['val'].count()
df.nlargest(100, 'col')
df.sample(n=1000, random_state=42)
df.pivot_table(values='val', index='col', aggfunc='mean')
```

### I/O Functions

#### CSV
```python
df = npd.read_csv('file.csv', sep=',', usecols=['col1', 'col2'], dtype={'id': 'int64'})
lf = npd.read_csv_lazy('file.csv', n_rows=1000)
```

#### Parquet
```python
df = npd.read_parquet('file.parquet', columns=['col1', 'col2'])
lf = npd.read_parquet_lazy('file.parquet')
```

#### Excel
```python
df = npd.read_excel('file.xlsx', sheet_name=0, usecols=['col1'], nrows=1000)
lf = npd.read_excel_lazy('file.xlsx', sheet_name='Sheet1')
```

#### JSON
```python
df = npd.read_json('file.json', dtype={'id': 'int64'})
lf = npd.read_json_lazy('file.json', lines=True)
```

## 🔄 Migration from pandas

```python
# Before
import pandas as pd
df = pd.read_csv('data.csv')
result = df.groupby('category')['value'].mean()

# After — same code, faster execution
import nitro_pandas as npd
df = npd.read_csv('data.csv')
result = df.groupby('category')['value'].mean()
```

Key differences to be aware of:

- **`df['col']`** returns a nitro-pandas `Series` (not a pandas Series) — it's compatible with boolean indexing and most pandas operations
- **No `inplace` parameter** — all operations return new DataFrames
- **No mixed column types** — each column must have a consistent type (Polars requirement)
- **Unimplemented methods** fall back to pandas automatically with a `PandasFallbackWarning`

```python
import warnings
from nitro_pandas import PandasFallbackWarning

# Silence fallback warnings if needed
warnings.filterwarnings("ignore", category=PandasFallbackWarning)
```

## 🏗️ Project Structure

```
nitro-pandas/
├── nitro_pandas/
│   ├── __init__.py      # Public API
│   ├── dataframe.py     # DataFrame, Series, GroupBy
│   ├── profiling.py     # profile_compare
│   ├── lazyframe.py     # LazyFrame
│   └── io/              # read_csv, read_parquet, read_excel, read_json
├── tests/
│   ├── test_dataframe.py
│   ├── test_profiling.py
│   ├── test_groupby.py
│   ├── test_io.py
│   └── test_runner.py
├── pyproject.toml
└── CHANGELOG.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/Wassim17Labdi/nitro-pandas.git
cd nitro-pandas
uv sync --dev
uv run python tests/test_runner.py
```

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Polars](https://www.pola.rs/) — For the high-performance DataFrame engine
- [pandas](https://pandas.pydata.org/) — For the API inspiration and fallback support

---

<div align="center">

**Made with ❤️ for the Python data science community**

⭐ Star this repo if you find it useful!

</div>
