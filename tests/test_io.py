"""
Tests for I/O operations (read/write).

This module contains comprehensive tests for nitro-pandas I/O functionality,
including reading and writing CSV, Parquet, JSON, and Excel files, both
eagerly and lazily. All tests verify that operations work correctly and
return nitro-pandas DataFrame or LazyFrame objects.
"""

import warnings
warnings.filterwarnings('ignore')
import nitro_pandas as npd
import pandas as pd
import tempfile
import os
import json
import polars as pl
try:
    from .helpers import create_sample_csv, create_sample_parquet, create_sample_excel, create_sample_json
except ImportError:
    from helpers import create_sample_csv, create_sample_parquet, create_sample_excel, create_sample_json


def test_read_csv():
    """
    Test CSV reading functionality.
    
    This test verifies that read_csv works correctly with various parameters
    including column selection, custom names, and null value handling.
    """
    csv_path = create_sample_csv()
    try:
        # 1. Basic reading
        df = npd.read_csv(csv_path)
        assert df.shape == (4, 4), "Basic read shape error"
        assert "name" in df.columns

        # 2. Specific parameters
        df = npd.read_csv(csv_path, usecols=["name", "value"], na_values="", comment_prefix="#")
        assert df.shape[1] == 2, "Column selection error"
        assert df.columns == ["name", "value"]

        # 3. Custom names
        df = npd.read_csv(csv_path, names=["ID", "NOM", "VALEUR", "DATE"], header=0)
        assert df.columns == ["ID", "NOM", "VALEUR", "DATE"]

        # 4. Head and tail
        assert df.head(2).shape[0] == 2
        assert df.tail(2).shape[0] == 2

        # 5. Conversion to pandas
        pandas_df = df.to_pandas()
        assert isinstance(pandas_df, pd.DataFrame)

        print("OK Tous les tests CSV sont passes avec succes.")

    finally:
        os.unlink(csv_path)


def test_read_parquet():
    """
    Test Parquet reading functionality.
    
    This test verifies that read_parquet works correctly with column selection
    and basic operations.
    """
    parquet_path = create_sample_parquet()
    try:
        # 1. Basic reading
        df = npd.read_parquet(parquet_path)
        assert df.shape == (4, 4), "Basic parquet read shape error"
        assert "name" in df.columns

        # 2. Reading with column selection
        df = npd.read_parquet(parquet_path, columns=["name", "value"])
        assert df.shape[1] == 2, "Parquet column selection error"
        assert df.columns == ["name", "value"]

        # 3. Head and tail
        assert df.head(2).shape[0] == 2
        assert df.tail(2).shape[0] == 2

        # 4. Conversion to pandas
        pandas_df = df.to_pandas()
        assert isinstance(pandas_df, pd.DataFrame)

        print("OK Tous les tests Parquet sont passes avec succes.")

    finally:
        os.unlink(parquet_path)


def test_read_excel():
    """
    Test Excel reading functionality.
    
    This test verifies that read_excel works correctly with column selection
    and basic operations.
    """
    excel_path = create_sample_excel()
    try:
        # 1. Basic reading
        df = npd.read_excel(excel_path)
        assert df.shape == (4, 4), "Basic excel read shape error"
        assert "name" in df.columns

        # 2. Reading with column selection
        df = npd.read_excel(excel_path, usecols=["name", "value"])
        assert df.shape[1] == 2, "Excel column selection error"
        assert df.columns == ["name", "value"]

        # 3. Head and tail
        assert df.head(2).shape[0] == 2
        assert df.tail(2).shape[0] == 2

        # 4. Conversion to pandas
        pandas_df = df.to_pandas()
        assert isinstance(pandas_df, pd.DataFrame)

        print("OK Tous les tests Excel sont passes avec succes.")

    finally:
        os.unlink(excel_path)


def test_read_json():
    """
    Test JSON reading functionality.
    
    This test verifies that read_json works correctly with schema specification
    and basic operations.
    """
    json_path = create_sample_json()
    try:
        # 1. Basic reading
        df = npd.read_json(json_path)
        assert df.shape == (4, 4), "Basic json read shape error"
        assert "name" in df.columns

        # 2. Reading with explicit schema
        df = npd.read_json(json_path, schema={"id": pl.Int64, "name": pl.Utf8, "value": pl.Float64, "date": pl.Utf8})
        assert df.shape == (4, 4), "Schema json read shape error"

        # 3. Head and tail
        assert df.head(2).shape[0] == 2
        assert df.tail(2).shape[0] == 2

        # 4. Conversion to pandas
        pandas_df = df.to_pandas()
        assert isinstance(pandas_df, pd.DataFrame)

        print("OK Tous les tests JSON sont passes avec succes.")

    finally:
        os.unlink(json_path)


def test_read_json_enhanced():
    """
    Test enhanced JSON reading parameters.
    
    This test verifies that read_json works correctly with dtype, n_rows,
    infer_schema_length, and schema parameters.
    """
    json_path = create_sample_json()
    try:
        df = npd.read_json(json_path, dtype={"id": pl.Int64, "value": pl.Float64})
        assert df.to_pandas()["id"].dtype == "int64", "dtype id error"
        assert df.to_pandas()["value"].dtype == "float64", "dtype value error"
        
        # Test with n_rows
        df_limited = npd.read_json(json_path, n_rows=2)
        assert df_limited.shape[0] == 2, "n_rows error"
        
        # Test with infer_schema_length
        df_infer = npd.read_json(json_path, infer_schema_length=2)
        assert df_infer.shape == (4, 4), "infer_schema_length error"
        
        # Test with complete schema
        df_schema = npd.read_json(json_path, schema={"id": pl.Int64, "name": pl.Utf8, "value": pl.Float64, "date": pl.Utf8})
        assert df_schema.to_pandas()["id"].dtype == "int64", "schema error"
        
        print("OK Test read_json enhanced OK")
    finally:
        os.unlink(json_path)
    
    # Test JSON Lines format
    json_lines_path = None
    try:
        json_lines_path = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl').name
        with open(json_lines_path, 'w') as f:
            f.write('{"id": 1, "name": "John", "value": 10.5}\n')
            f.write('{"id": 2, "name": "Jane", "value": 20.3}\n')
            f.write('{"id": 3, "name": "Bob", "value": 30.75}\n')
        
        df_lines = npd.read_json(json_lines_path, lines=True)
        assert df_lines.shape == (3, 3), "JSON Lines shape error"
        assert "name" in df_lines.columns, "JSON Lines columns error"
        
        # Test with n_rows on JSON Lines
        df_lines_limited = npd.read_json(json_lines_path, lines=True, n_rows=2)
        assert df_lines_limited.shape[0] == 2, "n_rows on JSON Lines error"
        
        print("OK Test read_json JSON Lines OK")
    finally:
        if json_lines_path and os.path.exists(json_lines_path):
            os.unlink(json_lines_path)


def test_read_csv_lazy():
    """
    Test lazy CSV reading functionality.
    
    This test verifies that read_csv_lazy works correctly and returns
    a LazyFrame that can be collected into a DataFrame.
    """
    csv_path = create_sample_csv()
    try:
        # Test 1: Basic lazy reading
        lf = npd.read_csv_lazy(csv_path)
        df = lf.collect()
        assert df.shape == (4, 4), "Lazy CSV shape error"
        assert "name" in df.columns
        assert isinstance(df.to_pandas(), pd.DataFrame)
        
        print("OK Test read_csv_lazy OK")
    finally:
        os.unlink(csv_path)


def test_read_csv_lazy_enhanced():
    """
    Test enhanced lazy CSV reading parameters.
    
    This test verifies that read_csv_lazy works correctly with various
    parameters including n_rows, null_values, and encoding.
    Note: scan_csv doesn't support column selection directly, so we test
    other parameters instead.
    """
    csv_path = create_sample_csv()
    try:
        # Note: scan_csv doesn't support columns parameter directly
        # We test other parameters instead
        
        # Test with n_rows
        lf = npd.read_csv_lazy(csv_path, n_rows=2)
        df = lf.collect()
        assert df.shape[0] == 2, "Lazy CSV n_rows error"
        
        # Test with null_values
        lf = npd.read_csv_lazy(csv_path, null_values="")
        df = lf.collect()
        assert df.shape == (4, 4), "Lazy CSV null_values error"
        
        # Test with encoding (Polars uses 'utf8' not 'utf-8')
        lf = npd.read_csv_lazy(csv_path, encoding="utf8")
        df = lf.collect()
        assert df.shape == (4, 4), "Lazy CSV encoding error"
        
        print("OK Test read_csv_lazy enhanced OK")
    finally:
        os.unlink(csv_path)


def test_write_csv_enhanced():
    """
    Test enhanced CSV writing functionality.
    
    This test verifies that DataFrame.to_csv works correctly with various
    parameters including custom separator, header control, and quote characters.
    """
    csv_path = create_sample_csv()
    out_path = csv_path + ".enhanced.csv"
    try:
        df = npd.read_csv(csv_path)
        
        # Test with custom separator using DataFrame.to_csv
        df.to_csv(out_path, separator=";")
        df_semicolon = npd.read_csv(out_path, sep=";")
        assert df_semicolon.shape == df.shape, "Shape mismatch with semicolon separator"
        
        # Test without header
        out_path_no_header = csv_path + ".no_header.csv"
        df.to_csv(out_path_no_header, include_header=False)
        df_no_header = npd.read_csv(out_path_no_header, header=None)
        assert df_no_header.shape == df.shape, "Shape mismatch without header"
        
        # Test with custom quote char
        out_path_quotes = csv_path + ".quotes.csv"
        df.to_csv(out_path_quotes, quote_char="'")
        df_quotes = npd.read_csv(out_path_quotes, truncate_ragged_lines=True)
        assert df_quotes.shape == df.shape, "Shape mismatch with custom quote char"
        
        print("OK Test to_csv enhanced OK")
    finally:
        os.unlink(csv_path)
        for ext in [".enhanced.csv", ".no_header.csv", ".quotes.csv"]:
            out = csv_path + ext
            if os.path.exists(out):
                os.unlink(out)


def test_read_parquet_enhanced():
    """
    Test enhanced Parquet reading parameters.
    
    This test verifies that read_parquet works correctly with dtype, n_rows,
    columns, and parallel parameters.
    """
    parquet_path = create_sample_parquet()
    try:
        # Test 1: Reading with dtype
        df = npd.read_parquet(parquet_path, dtype={"id": pl.Int64, "value": pl.Float64})
        assert df.to_pandas()["id"].dtype == "int64", "dtype id error"
        assert df.to_pandas()["value"].dtype == "float64", "dtype value error"
        
        # Test 2: Reading with n_rows
        df_limited = npd.read_parquet(parquet_path, n_rows=2)
        assert df_limited.shape[0] == 2, "n_rows error"
        
        # Test 3: Reading with columns
        df_columns = npd.read_parquet(parquet_path, columns=["id", "name"])
        assert df_columns.shape[1] == 2, "columns error"
        assert "id" in df_columns.columns and "name" in df_columns.columns, "columns selection error"
        
        # Test 4: Reading with columns + n_rows
        df_combo = npd.read_parquet(parquet_path, columns=["id", "name"], n_rows=2)
        assert df_combo.shape == (2, 2), "columns + n_rows error"
        
        # Test 5: Reading with parallel (API compatibility, Polars handles automatically)
        df_parallel = npd.read_parquet(parquet_path, parallel="auto")
        assert df_parallel.shape == (4, 4), "parallel error"
        
        # Test 6: Reading with dtype + n_rows
        df_dtype_nrows = npd.read_parquet(parquet_path, dtype={"id": pl.Int64}, n_rows=2)
        assert df_dtype_nrows.shape[0] == 2, "dtype + n_rows error"
        assert df_dtype_nrows.to_pandas()["id"].dtype == "int64", "dtype with n_rows error"
        
        print("OK Test read_parquet enhanced OK")
    finally:
        os.unlink(parquet_path)


def test_read_parquet_lazy():
    """
    Test lazy Parquet reading functionality.
    
    This test verifies that read_parquet_lazy works correctly with various
    parameters including dtype, n_rows, columns, and parallel.
    """
    parquet_path = create_sample_parquet()
    try:
        # Test 1: Basic lazy reading
        df_lazy = npd.read_parquet_lazy(parquet_path)
        df = df_lazy.collect()
        assert df.shape == (4, 4), "Lazy parquet shape error"
        assert "name" in df.columns
        assert isinstance(df.to_pandas(), pd.DataFrame)
        
        # Test 2: Reading with dtype
        lf_dtype = npd.read_parquet_lazy(parquet_path, dtype={"id": pl.Int64})
        df_dtype = lf_dtype.collect()
        assert df_dtype.to_pandas()["id"].dtype == "int64", "Lazy parquet dtype error"
        
        # Test 3: Reading with n_rows
        lf_nrows = npd.read_parquet_lazy(parquet_path, n_rows=2)
        df_nrows = lf_nrows.collect()
        assert df_nrows.shape[0] == 2, "Lazy parquet n_rows error"
        
        # Test 4: Reading with columns
        lf_columns = npd.read_parquet_lazy(parquet_path, columns=["id", "name"])
        df_columns = lf_columns.collect()
        assert df_columns.shape[1] == 2, "Lazy parquet columns error"
        assert "id" in df_columns.columns and "name" in df_columns.columns, "Lazy columns selection error"
        
        # Test 5: Reading with columns + n_rows
        lf_combo = npd.read_parquet_lazy(parquet_path, columns=["id", "name"], n_rows=2)
        df_combo = lf_combo.collect()
        assert df_combo.shape == (2, 2), "Lazy parquet columns + n_rows error"
        
        # Test 6: Reading with parallel (API compatibility)
        lf_parallel = npd.read_parquet_lazy(parquet_path, parallel="auto")
        df_parallel = lf_parallel.collect()
        assert df_parallel.shape == (4, 4), "Lazy parquet parallel error"
        
        # Test 7: Reading with dtype + n_rows
        lf_dtype_nrows = npd.read_parquet_lazy(parquet_path, dtype={"id": pl.Int64}, n_rows=2)
        df_dtype_nrows = lf_dtype_nrows.collect()
        assert df_dtype_nrows.shape[0] == 2, "Lazy parquet dtype + n_rows error"
        assert df_dtype_nrows.to_pandas()["id"].dtype == "int64", "Lazy dtype with n_rows error"
        
        print("OK Test read_parquet_lazy OK")
    finally:
        os.unlink(parquet_path)


def test_to_csv():
    """
    Test basic CSV writing functionality.
    
    This test verifies that DataFrame.to_csv works correctly and produces
    a file that can be read back with the same shape and columns.
    """
    csv_path = create_sample_csv()
    out_path = csv_path + ".out.csv"
    try:
        df = npd.read_csv(csv_path)
        df.to_csv(out_path)
        df2 = npd.read_csv(out_path)
        assert df2.shape == df.shape, "Shape mismatch after to_csv"
        assert df2.columns == df.columns, "Columns mismatch after to_csv"
        print("OK Test to_csv OK")
    finally:
        os.unlink(csv_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_to_parquet():
    """
    Test basic Parquet writing functionality.
    
    This test verifies that DataFrame.to_parquet works correctly and produces
    a file that can be read back with the same shape and columns.
    """
    csv_path = create_sample_csv()
    parquet_path = csv_path + ".out.parquet"
    try:
        df = npd.read_csv(csv_path)
        df.to_parquet(parquet_path)
        df2 = npd.read_parquet(parquet_path)
        assert df2.shape == df.shape, "Shape mismatch after to_parquet"
        assert df2.columns == df.columns, "Columns mismatch after to_parquet"
        print("OK Test to_parquet OK")
    finally:
        os.unlink(csv_path)
        if os.path.exists(parquet_path):
            os.unlink(parquet_path)


def test_to_json():
    """
    Test basic JSON writing functionality.
    
    This test verifies that DataFrame.to_json works correctly and produces
    a file that can be read back with the same shape and columns.
    """
    csv_path = create_sample_csv()
    json_path = csv_path + ".out.json"
    try:
        df = npd.read_csv(csv_path)
        df.to_json(json_path)
        df2 = npd.read_json(json_path)
        assert df2.shape == df.shape, "Shape mismatch after to_json"
        assert df2.columns == df.columns, "Columns mismatch after to_json"
        print("OK Test to_json OK")
    finally:
        os.unlink(csv_path)
        if os.path.exists(json_path):
            os.unlink(json_path)


def test_to_excel():
    """
    Test basic Excel writing functionality.
    
    This test verifies that DataFrame.to_excel works correctly and produces
    a file that can be read back with the same shape and columns.
    """
    csv_path = create_sample_csv()
    excel_path = csv_path + ".out.xlsx"
    try:
        df = npd.read_csv(csv_path)
        df.to_excel(excel_path)
        df2 = npd.read_excel(excel_path)
        assert df2.shape == df.shape, "Shape mismatch after to_excel"
        assert df2.columns == df.columns, "Columns mismatch after to_excel"
        print("OK Test to_excel OK")
    finally:
        os.unlink(csv_path)
        if os.path.exists(excel_path):
            os.unlink(excel_path)


def test_read_json_lazy_new():
    """
    Test lazy JSON reading functionality.
    
    This test verifies that read_json_lazy works correctly with various
    parameters including dtype, n_rows, schema, and infer_schema_length.
    """
    json_path = create_sample_json()
    try:
        # Test 1: Basic lazy reading
        lf = npd.read_json_lazy(json_path)
        df = lf.collect()
        assert df.shape == (4, 4), "Lazy json shape error"
        assert "name" in df.columns, "Lazy json columns error"
        
        # Test 2: Reading with dtype
        lf_dtype = npd.read_json_lazy(json_path, dtype={"id": pl.Int64})
        df_dtype = lf_dtype.collect()
        assert df_dtype.to_pandas()["id"].dtype == "int64", "Lazy json dtype error"
        
        # Test 3: Reading with n_rows
        lf_nrows = npd.read_json_lazy(json_path, n_rows=2)
        df_nrows = lf_nrows.collect()
        assert df_nrows.shape[0] == 2, "Lazy json n_rows error"
        
        # Test 4: Reading with schema
        lf_schema = npd.read_json_lazy(json_path, schema={"id": pl.Int64, "name": pl.Utf8, "value": pl.Float64, "date": pl.Utf8})
        df_schema = lf_schema.collect()
        assert df_schema.shape == (4, 4), "Lazy json schema error"
        
        # Test 5: Reading with infer_schema_length
        lf_infer = npd.read_json_lazy(json_path, infer_schema_length=2)
        df_infer = lf_infer.collect()
        assert df_infer.shape == (4, 4), "Lazy json infer_schema_length error"
        
        print("OK Test read_json_lazy OK")
    finally:
        os.unlink(json_path)
    
    # Test 6: JSON Lines with lazy
    json_lines_path = None
    try:
        json_lines_path = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl').name
        with open(json_lines_path, 'w') as f:
            f.write('{"id": 1, "name": "John", "value": 10.5}\n')
            f.write('{"id": 2, "name": "Jane", "value": 20.3}\n')
            f.write('{"id": 3, "name": "Bob", "value": 30.75}\n')
        
        lf_lines = npd.read_json_lazy(json_lines_path, lines=True)
        df_lines = lf_lines.collect()
        assert df_lines.shape == (3, 3), "Lazy JSON Lines shape error"
        assert "name" in df_lines.columns, "Lazy JSON Lines columns error"
        
        # Test with n_rows on JSON Lines lazy
        lf_lines_limited = npd.read_json_lazy(json_lines_path, lines=True, n_rows=2)
        df_lines_limited = lf_lines_limited.collect()
        assert df_lines_limited.shape[0] == 2, "n_rows on lazy JSON Lines error"
        
        print("OK Test read_json_lazy JSON Lines OK")
    finally:
        if json_lines_path and os.path.exists(json_lines_path):
            os.unlink(json_lines_path)


def test_read_excel_lazy_new():
    """
    Test basic lazy Excel reading functionality.
    
    This test verifies that read_excel_lazy works correctly and returns
    a LazyFrame that can be collected into a DataFrame.
    """
    excel_path = create_sample_excel()
    try:
        lf = npd.read_excel_lazy(excel_path)
        df = lf.collect()
        assert df.shape == (4, 4)
        assert "name" in df.columns
        print("OK Test read_excel_lazy OK")
    finally:
        os.unlink(excel_path)


def test_read_excel_lazy_enhanced():
    """
    Test enhanced lazy Excel reading parameters.
    
    This test verifies that read_excel_lazy works correctly with usecols,
    nrows, skiprows, and names parameters.
    """
    excel_path = create_sample_excel()
    try:
        # Test with column selection
        lf = npd.read_excel_lazy(excel_path, usecols=["name", "value"])
        df = lf.collect()
        assert df.shape[1] == 2, "Lazy excel column selection shape error"
        assert df.columns == ["name", "value"]
        
        # Test with nrows limit
        lf = npd.read_excel_lazy(excel_path, nrows=2)
        df = lf.collect()
        assert df.shape[0] == 2, "Lazy excel nrows error"
        
        # Test with skiprows
        lf = npd.read_excel_lazy(excel_path, skiprows=1)
        df = lf.collect()
        assert df.shape[0] == 3, "Lazy excel skiprows error"
        
        # Test with custom names
        lf = npd.read_excel_lazy(excel_path, names=["ID", "NOM", "VALEUR", "DATE"])
        df = lf.collect()
        assert df.columns == ["ID", "NOM", "VALEUR", "DATE"], "Lazy excel names error"
        
        print("OK Test read_excel_lazy enhanced OK")
    finally:
        os.unlink(excel_path)


def test_dtype_casting():
    """
    Test dtype casting functionality across all I/O formats.
    
    This test verifies that dtype parameter works correctly for CSV, Excel,
    Parquet, and lazy reading functions.
    """
    csv_path = create_sample_csv()
    try:
        # Test CSV with dtype casting (only id column to avoid comma issue)
        df = npd.read_csv(csv_path, dtype={"id": pl.Int64})
        assert df.to_pandas()["id"].dtype == "int64", "CSV dtype casting failed for id"
        
        # Test Excel with dtype casting
        excel_path = create_sample_excel()
        try:
            df_excel = npd.read_excel(excel_path, dtype={"id": pl.Int64, "value": pl.Float64})
            assert df_excel.to_pandas()["id"].dtype == "int64", "Excel dtype casting failed for id"
            assert df_excel.to_pandas()["value"].dtype == "float64", "Excel dtype casting failed for value"
        finally:
            os.unlink(excel_path)
        
        # Test Parquet with dtype casting
        parquet_path = create_sample_parquet()
        try:
            df_parquet = npd.read_parquet(parquet_path, dtype={"id": pl.Int64, "value": pl.Float64})
            assert df_parquet.to_pandas()["id"].dtype == "int64", "Parquet dtype casting failed for id"
            assert df_parquet.to_pandas()["value"].dtype == "float64", "Parquet dtype casting failed for value"
        finally:
            os.unlink(parquet_path)
        
        # Test lazy functions with dtype
        df_lazy = npd.read_csv_lazy(csv_path, dtype={"id": pl.Int64})
        df_collected = df_lazy.collect()
        assert df_collected.to_pandas()["id"].dtype == "int64", "Lazy CSV dtype casting failed"
        
        print("OK Test dtype casting OK")
    finally:
        os.unlink(csv_path)
