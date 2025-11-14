"""
Tests for DataFrame operations.

This module contains comprehensive tests for nitro-pandas DataFrame functionality,
including indexing, filtering, transformations, and pandas compatibility.
"""

import warnings
warnings.filterwarnings('ignore')
import nitro_pandas as npd
import pandas as pd
import tempfile
import os
import polars as pl
try:
    from .helpers import create_sample_csv
except ImportError:
    from helpers import create_sample_csv


def test_dataframe_creation():
    """
    Test direct DataFrame creation without requiring Polars DataFrame.
    
    This test verifies that DataFrame can be created from various data sources
    (dict, Polars DataFrame, empty) with pandas-like syntax.
    """
    # Test 1: Creation from dict (pandas-like syntax)
    df1 = npd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert isinstance(df1, npd.DataFrame), "DataFrame() from dict must return a DataFrame"
    assert df1.shape == (3, 2), "DataFrame() from dict must have correct shape"
    assert 'a' in df1.columns and 'b' in df1.columns, "DataFrame() from dict must have correct columns"
    
    # Test 2: Creation from Polars DataFrame (backward compatibility)
    pl_df = pl.DataFrame({'x': [1, 2], 'y': [3, 4]})
    df2 = npd.DataFrame(pl_df)
    assert isinstance(df2, npd.DataFrame), "DataFrame() from Polars DataFrame must work"
    assert df2.shape == (2, 2), "DataFrame() from Polars DataFrame must have correct shape"
    
    # Test 3: Creation from dict with explicit data parameter
    df3 = npd.DataFrame(data={'ville': ['Paris', 'Lyon'], 'pop': [2000000, 500000]})
    assert isinstance(df3, npd.DataFrame), "DataFrame(data=dict) must work"
    assert df3.shape == (2, 2), "DataFrame(data=dict) must have correct shape"
    
    # Test 4: Empty DataFrame creation
    df4 = npd.DataFrame()
    assert isinstance(df4, npd.DataFrame), "Empty DataFrame() must work"
    assert df4.shape == (0, 0), "Empty DataFrame() must have shape (0, 0)"
    
    print("OK Test DataFrame creation OK")


def test_query_method():
    """
    Test query() method for filtering DataFrames with string expressions.
    
    This test verifies that pandas-like query strings are correctly converted
    to Polars expressions and applied to both DataFrame and LazyFrame.
    """
    import tempfile
    csv_content = """id,cat,val
1,A,10
2,B,20
3,A,30
4,B,40
5,A,50
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(csv_content)
        csv_path = f.name
    try:
        # Test DataFrame query
        df = npd.read_csv(csv_path)
        filtered = df.query("val > 20 and cat == 'A'")
        values = filtered.to_pandas()["val"].tolist()
        assert values == [30, 50], "Query must filter correctly"
        print("OK Test DataFrame .query() OK")

        # Test LazyFrame query
        lf = npd.read_csv_lazy(csv_path)
        filtered_lf = lf.query("val > 20 and cat == 'A'")
        df_lazy = filtered_lf.collect()
        values_lazy = df_lazy.to_pandas()["val"].tolist()
        assert values_lazy == [30, 50], "LazyFrame query must filter correctly"
        print("OK Test LazyFrame .query() OK")
    finally:
        os.unlink(csv_path)


def test_loc_iloc():
    """
    Test loc and iloc indexers for label-based and position-based indexing.
    
    This test verifies that both loc (label-based) and iloc (position-based)
    indexing work correctly with various selection patterns.
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        # Test loc: label-based selection
        assert df.loc[0, "name"] == "John", "loc must select by label"
        assert df.loc[1:3, ["id", "value"].copy()].shape == (3, 2), "loc must support slice and column selection"
        assert df.loc[:, "id"].to_list() == [1, 2, 3, 4], "loc must select entire column"
        # Test iloc: position-based selection
        assert df.iloc[0, 1] == "John", "iloc must select by position"
        assert df.iloc[1:3, 0:2].shape == (2, 2), "iloc must support slice selection"
        assert df.iloc[:, 0].to_list() == [1, 2, 3, 4], "iloc must select entire column"
        print("OK Test loc/iloc OK")
    finally:
        os.unlink(csv_path)


def test_loc_mask():
    """
    Test loc with pandas Series boolean mask.
    
    This test verifies that df.loc[mask] works correctly when mask is
    a pandas Series boolean (from df['col'] > value).
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        # Create pandas Series boolean mask
        mask = df["id"] > 2
        filtered = df.loc[mask]
        assert filtered.shape[0] == 2, "loc[mask] must filter correctly"
        assert filtered.to_pandas()["name"].tolist() == ["Bob", "Alice"], "loc[mask] must return correct rows"
        print("OK Test loc[mask] OK")
    finally:
        os.unlink(csv_path)


def test_loc_mask_dataframe():
    """
    Test loc with pandas DataFrame boolean mask.
    
    This test verifies that df.loc[df > 2] works correctly when the mask
    is a pandas DataFrame boolean (from df > value).
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        # Create pandas DataFrame boolean mask
        mask_df = df > 2  # Returns pandas DataFrame boolean
        filtered = df.loc[mask_df]
        # Verify result is correct
        assert isinstance(filtered, npd.DataFrame), "df.loc[df > 2] must return a DataFrame nitro-pandas"
        # Rows with id > 2 should be included
        pdf = filtered.to_pandas()
        assert all(pdf["id"] > 2), "df.loc[df > 2] must filter rows where id > 2"
        print("OK Test loc[mask DataFrame] OK")
    finally:
        os.unlink(csv_path)


def test_loc_direct_comparison():
    """
    Test loc with direct comparison expression (no intermediate variable).
    
    This test verifies that df.loc[df > 2] works directly without declaring
    an intermediate mask variable, testing the full expression evaluation chain.
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        # Direct test: df.loc[df > 2] without declaring mask variable
        filtered = df.loc[df > 2]
        # Verify result is correct
        assert isinstance(filtered, npd.DataFrame), "df.loc[df > 2] must return a DataFrame nitro-pandas"
        # Rows with id > 2 should be included
        pdf = filtered.to_pandas()
        assert len(pdf) > 0, "df.loc[df > 2] must return at least one row"
        assert all(pdf["id"] > 2), f"df.loc[df > 2] must return only rows with id > 2, got: {pdf['id'].tolist()}"
        print("OK Test loc[df > 2] direct OK")
    finally:
        os.unlink(csv_path)


def test_loc_mask_col():
    """
    Test loc with boolean mask and column selection.
    
    This test verifies that df.loc[mask, column] works correctly,
    combining boolean filtering with column selection.
    """
    import tempfile
    csv_content = """id,cat,val
1,A,10
2,B,20
3,A,30
4,B,40
5,A,50
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(csv_content)
        csv_path = f.name
    try:
        df = npd.read_csv(csv_path)
        # Create pandas Series boolean mask
        mask = df["cat"] == "A"
        # Select rows with mask and specific column
        result = df.loc[mask, "val"]
        # Expected: Polars Series with [10, 30, 50]
        assert result.to_list() == [10, 30, 50], f"Result: {result.to_list()}"
        print("OK Test loc[mask, col] OK")
    finally:
        os.unlink(csv_path)


def test_direct_mask():
    """
    Test direct boolean masking with df[mask] syntax.
    
    This test verifies that df[mask] works correctly for boolean filtering
    and column selection, similar to pandas behavior.
    """
    import tempfile
    csv_content = """id,cat,val
1,A,10
2,B,20
3,A,30
4,B,40
5,A,50
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(csv_content)
        csv_path = f.name
    try:
        df = npd.read_csv(csv_path)
        # Test 1: df[mask] - boolean filtering
        filtered = df[df["val"] > 20]
        # Expected: DataFrame with rows 3, 4, 5
        assert isinstance(filtered, npd.DataFrame), "df[mask] must return a DataFrame nitro-pandas"
        pdf = filtered.to_pandas()
        assert pdf["id"].tolist() == [3, 4, 5], f"Result: {pdf['id'].tolist()}"
        assert pdf["val"].tolist() == [30, 40, 50], f"Result: {pdf['val'].tolist()}"
        
        # Test 2: df[['col1', 'col2']] - column selection
        selected = df[['id', 'val']]
        assert isinstance(selected, npd.DataFrame), "df[['col1', 'col2']] must return a DataFrame nitro-pandas"
        assert selected.shape[1] == 2, "df[['col1', 'col2']] must have 2 columns"
        assert 'id' in selected.columns and 'val' in selected.columns, "df[['col1', 'col2']] must have correct columns"
        
        # Test 3: df['col'] - single column selection (returns pandas Series for pandas expressions)
        single_col = df['val']
        import pandas as pd
        assert isinstance(single_col, pd.Series), "df['col'] must return a pandas Series (for pandas expressions)"
        
        print("OK Test direct mask df[mask] OK")
    finally:
        os.unlink(csv_path)


def test_dataframe_slicing():
    """
    Test pandas-like slicing: df[:10], df[5:10], df[::2], etc.
    
    This test verifies that DataFrame slicing works correctly with various
    slice patterns, returning nitro-pandas DataFrame objects.
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        
        # Test 1: df[:10] - first 10 rows
        result1 = df[:10]
        assert isinstance(result1, npd.DataFrame), "df[:10] must return a DataFrame nitro-pandas"
        assert result1.shape[0] <= 10, "df[:10] must have at most 10 rows"
        
        # Test 2: df[5:10] - rows 5 to 9
        if df.shape[0] >= 10:
            result2 = df[5:10]
            assert isinstance(result2, npd.DataFrame), "df[5:10] must return a DataFrame nitro-pandas"
            assert result2.shape[0] == 5, "df[5:10] must have 5 rows"
        
        # Test 3: df[::2] - every other row
        result3 = df[::2]
        assert isinstance(result3, npd.DataFrame), "df[::2] must return a DataFrame nitro-pandas"
        assert result3.shape[0] <= df.shape[0], "df[::2] must have fewer or equal rows"
        
        # Test 4: df[-5:] - last 5 rows
        if df.shape[0] >= 5:
            result4 = df[-5:]
            assert isinstance(result4, npd.DataFrame), "df[-5:] must return a DataFrame nitro-pandas"
            assert result4.shape[0] == 5, "df[-5:] must have 5 rows"
        
        # Test 5: df[1:3] - rows 1 to 2
        result5 = df[1:3]
        assert isinstance(result5, npd.DataFrame), "df[1:3] must return a DataFrame nitro-pandas"
        assert result5.shape[0] <= 2, "df[1:3] must have at most 2 rows"
        
        # Verify columns are preserved
        assert result1.shape[1] == df.shape[1], "df[:10] must keep all columns"
        assert result2.shape[1] == df.shape[1] if df.shape[0] >= 10 else True, "df[5:10] must keep all columns"
        
        print("OK Test DataFrame slicing OK")
    finally:
        os.unlink(csv_path)


def test_pandas_fallback_describe():
    """
    Test pandas fallback for unimplemented methods.
    
    This test verifies that methods not explicitly implemented in nitro-pandas
    fall back to pandas, returning pandas objects directly.
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)

        # describe() is not implemented, so it falls back to pandas
        desc = df.describe()
        # Fallback pandas returns pandas.DataFrame directly
        assert isinstance(desc, pd.DataFrame), "describe must return pandas.DataFrame (pandas fallback)"
        # Sanity checks: columns/number of rows > 0
        assert desc.shape[0] > 0 and desc.shape[1] > 0, "describe must return statistics"
        print("OK Test pandas fallback (describe) renvoie pandas.DataFrame")
    finally:
        os.unlink(csv_path)


def test_pandas_fallback_module_level():
    """
    Test pandas fallback at module level for unimplemented functions.
    
    This test verifies that pandas functions at the package level are accessible
    via nitro-pandas module, falling back to pandas when not implemented.
    """
    # Test 1: npd.get_dummies() - pandas function at module level
    df_pd = pd.DataFrame({
        'a': [1, 2, 1, 2],
        'b': ['x', 'y', 'x', 'z']
    })
    result_dummies = npd.get_dummies(df_pd, columns=['b'])
    assert isinstance(result_dummies, pd.DataFrame), "npd.get_dummies() must return pandas.DataFrame"
    assert 'b_x' in result_dummies.columns, "npd.get_dummies() must create dummy columns"
    assert 'b_y' in result_dummies.columns, "npd.get_dummies() must create dummy columns"
    assert 'b_z' in result_dummies.columns, "npd.get_dummies() must create dummy columns"
    
    # Test 2: npd.date_range() - pandas function at module level
    dates = npd.date_range('2024-01-01', periods=5, freq='D')
    assert hasattr(dates, 'tolist'), "npd.date_range() must return DatetimeIndex-like object"
    assert len(dates) == 5, "npd.date_range() must return correct number of dates"
    
    # Test 3: npd.Timestamp() - pandas function at module level
    ts = npd.Timestamp('2024-01-01')
    assert isinstance(ts, pd.Timestamp), "npd.Timestamp() must return pandas.Timestamp"
    assert str(ts).startswith('2024-01-01'), "npd.Timestamp() must create correct timestamp"
    
    # Test 4: npd.NA - pandas constant at module level
    na_value = npd.NA
    assert na_value is pd.NA, "npd.NA must be pandas.NA"
    
    # Test 5: Verify that npd.get_dummies is actually pandas.get_dummies
    assert npd.get_dummies is pd.get_dummies, "npd.get_dummies must be pandas.get_dummies (fallback)"
    
    # Test 6: Verify that non-existent function raises AttributeError
    try:
        _ = npd.non_existent_function
        assert False, "Non-existent function should raise AttributeError"
    except AttributeError:
        pass  # Expected
    
    print("OK Test pandas fallback module level OK")


def test_sort_values_and_rename_and_drop_and_astype_and_fillna():
    """
    Test multiple DataFrame transformation methods.
    
    This test verifies that sort_values, rename, drop, astype, and fillna
    all work correctly and return nitro-pandas DataFrame objects.
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        # sort_values by id ascending -> first row id=1
        sorted_df = df.sort_values("id", ascending=True)
        assert sorted_df.to_pandas().iloc[0]["id"] == 1, "sort_values must sort correctly"
        # rename column name -> NAME
        renamed = df.rename(columns={"name": "NAME"})
        assert "NAME" in renamed.columns, "rename must rename columns correctly"
        # drop column date (with axis=1)
        dropped = df.drop(labels=["date"], axis=1)
        assert "date" not in dropped.columns, "drop must remove columns correctly"
        # astype: cast id to str (pandas type)
        casted = df.astype({"id": "str"})
        assert casted.to_pandas()["id"].dtype == object, "astype must cast to string correctly"
        # astype: cast id to int (pandas type)
        casted_int = df.astype({"id": int})
        assert casted_int.to_pandas()["id"].dtype in ["int64", "int32"], "astype must cast to int correctly"
        # astype: cast id to float (pandas type) - avoid value column as it contains formatted commas
        casted_float = df.astype({"id": float})
        assert "float" in str(casted_float.to_pandas()["id"].dtype), "astype must cast to float correctly"
        # fillna: fill missing value on value column -> 0
        filled = df.fillna({"value": 0})
        vals = filled.to_pandas()["value"].tolist()
        assert any(v == 0 or v == "0" for v in vals), "fillna must fill null values correctly"
        print("OK Test sort/rename/drop/astype/fillna OK")
    finally:
        os.unlink(csv_path)


def test_drop_with_axis():
    """
    Test drop() method with axis parameter.
    
    This test verifies that drop() correctly handles both row (axis=0)
    and column (axis=1) dropping with various label types.
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        
        # Test 1: drop rows with axis=0
        df_rows = df.drop(labels=[0, 1], axis=0)
        assert isinstance(df_rows, npd.DataFrame), "drop with axis=0 must return a DataFrame nitro-pandas"
        assert df_rows.shape[0] == df.shape[0] - 2, "drop with axis=0 must remove specified rows"
        assert df_rows.shape[1] == df.shape[1], "drop with axis=0 must keep all columns"
        
        # Test 2: drop columns with axis=1
        df_cols = df.drop(labels=["date"], axis=1)
        assert isinstance(df_cols, npd.DataFrame), "drop with axis=1 must return a DataFrame nitro-pandas"
        assert df_cols.shape[1] == df.shape[1] - 1, "drop with axis=1 must remove specified columns"
        assert df_cols.shape[0] == df.shape[0], "drop with axis=1 must keep all rows"
        assert "date" not in df_cols.columns, "drop with axis=1 must remove the 'date' column"
        
        # Test 3: drop multiple columns with axis=1
        df_multi = df.drop(labels=["date", "name"], axis=1)
        assert df_multi.shape[1] == df.shape[1] - 2, "drop with axis=1 must remove multiple columns"
        assert "date" not in df_multi.columns and "name" not in df_multi.columns
        
        # Test 4: drop single row with axis=0
        df_single = df.drop(labels=[0], axis=0)
        assert df_single.shape[0] == df.shape[0] - 1, "drop with axis=0 must remove one row"
        
        print("OK Test drop with axis OK")
    finally:
        os.unlink(csv_path)


def test_drop_duplicates_and_value_counts_and_reset_index():
    """
    Test drop_duplicates, value_counts, and reset_index methods.
    
    This test verifies that these data cleaning methods work correctly
    and return nitro-pandas DataFrame objects.
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        # drop_duplicates
        unique = df.drop_duplicates()
        assert isinstance(unique, npd.DataFrame), "drop_duplicates must return DataFrame"
        # value_counts
        counts = df.value_counts("name", sort=True)
        assert isinstance(counts, npd.DataFrame), "value_counts must return DataFrame"
        # reset_index
        reset = df.reset_index(drop=True)
        assert isinstance(reset, npd.DataFrame), "reset_index must return DataFrame"
        print("OK Test drop_duplicates/value_counts/reset_index OK")
    finally:
        os.unlink(csv_path)


def test_merge_and_concat_and_isna_notna():
    """
    Test merge, concat, isna, and notna methods.
    
    This test verifies that these data combination and null-checking
    methods work correctly and return nitro-pandas DataFrame objects.
    """
    csv_path = create_sample_csv()
    try:
        df1 = npd.read_csv(csv_path)
        df2 = npd.DataFrame({'id': [1, 2], 'extra': [100, 200]})
        # merge
        merged = df1.merge(df2, on='id', how='inner')
        assert isinstance(merged, npd.DataFrame), "merge must return DataFrame"
        # concat
        concatenated = npd.concat([df1, df1], axis=0)
        assert isinstance(concatenated, npd.DataFrame), "concat must return DataFrame"
        # isna
        nulls = df1.isna()
        assert isinstance(nulls, npd.DataFrame), "isna must return DataFrame"
        # notna
        not_nulls = df1.notna()
        assert isinstance(not_nulls, npd.DataFrame), "notna must return DataFrame"
        print("OK Test merge/concat/isna/notna OK")
    finally:
        os.unlink(csv_path)


def test_concat_content():
    """
    Test concat method with content verification for both axis=0 and axis=1.
    
    This test verifies that concat correctly concatenates DataFrames:
    - axis=0: vertical concatenation (row-wise), stacking rows
    - axis=1: horizontal concatenation (column-wise), adding columns
    """
    # Create test DataFrames
    df1 = npd.DataFrame({
        'id': [1, 2],
        'name': ['Alice', 'Bob'],
        'value': [10, 20]
    })
    
    df2 = npd.DataFrame({
        'id': [3, 4],
        'name': ['Charlie', 'David'],
        'value': [30, 40]
    })
    
    df3 = npd.DataFrame({
        'extra_col': ['X', 'Y'],
        'score': [100, 200]
    })
    
    # Test axis=0: vertical concatenation (row-wise)
    result_axis0 = npd.concat([df1, df2], axis=0)
    assert isinstance(result_axis0, npd.DataFrame), "concat with axis=0 must return DataFrame"
    assert result_axis0.shape == (4, 3), f"concat axis=0 must have 4 rows and 3 columns, got {result_axis0.shape}"
    
    # Verify content for axis=0
    pdf_axis0 = result_axis0.to_pandas()
    assert pdf_axis0['id'].tolist() == [1, 2, 3, 4], f"concat axis=0 must stack rows correctly, got {pdf_axis0['id'].tolist()}"
    assert pdf_axis0['name'].tolist() == ['Alice', 'Bob', 'Charlie', 'David'], f"concat axis=0 must preserve name column, got {pdf_axis0['name'].tolist()}"
    assert pdf_axis0['value'].tolist() == [10, 20, 30, 40], f"concat axis=0 must preserve value column, got {pdf_axis0['value'].tolist()}"
    
    # Test axis=1: horizontal concatenation (column-wise)
    result_axis1 = npd.concat([df1, df3], axis=1)
    assert isinstance(result_axis1, npd.DataFrame), "concat with axis=1 must return DataFrame"
    assert result_axis1.shape == (2, 5), f"concat axis=1 must have 2 rows and 5 columns, got {result_axis1.shape}"
    
    # Verify content for axis=1
    pdf_axis1 = result_axis1.to_pandas()
    # Check original columns from df1
    assert pdf_axis1['id'].tolist() == [1, 2], f"concat axis=1 must preserve df1 id column, got {pdf_axis1['id'].tolist()}"
    assert pdf_axis1['name'].tolist() == ['Alice', 'Bob'], f"concat axis=1 must preserve df1 name column, got {pdf_axis1['name'].tolist()}"
    assert pdf_axis1['value'].tolist() == [10, 20], f"concat axis=1 must preserve df1 value column, got {pdf_axis1['value'].tolist()}"
    # Check new columns from df3
    assert pdf_axis1['extra_col'].tolist() == ['X', 'Y'], f"concat axis=1 must add df3 extra_col column, got {pdf_axis1['extra_col'].tolist()}"
    assert pdf_axis1['score'].tolist() == [100, 200], f"concat axis=1 must add df3 score column, got {pdf_axis1['score'].tolist()}"
    
    # Verify all columns are present
    expected_cols_axis1 = ['id', 'name', 'value', 'extra_col', 'score']
    assert set(pdf_axis1.columns) == set(expected_cols_axis1), f"concat axis=1 must have all columns, got {list(pdf_axis1.columns)}"
    
    print("OK Test concat content with axis=0 and axis=1 OK")


def test_setitem_add_column():
    """
    Test __setitem__ method for adding/modifying columns with df['col'] = value.
    
    This test verifies that column assignment works correctly with:
    - Scalar values
    - Lists/arrays
    - Polars expressions
    - Pandas Series
    - Modifying existing columns
    """
    import polars as pl
    import pandas as pd
    
    # Test 1: df['new_col'] = scalar value
    df1 = npd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, 20, 30, 40]
    })
    df1['constant'] = 100
    assert "constant" in df1.columns, "df['col'] = scalar must add new column"
    assert df1.shape == (4, 3), "df['col'] = scalar must keep all columns"
    pdf1 = df1.to_pandas()
    assert pdf1['constant'].tolist() == [100, 100, 100, 100], "df['col'] = scalar must fill all rows"
    
    # Test 2: df['new_col'] = list
    df2 = npd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, 20, 30, 40]
    })
    df2['scores'] = [100, 200, 300, 400]
    assert "scores" in df2.columns, "df['col'] = list must add new column"
    assert df2.shape == (4, 3), "df['col'] = list must add column"
    pdf2 = df2.to_pandas()
    assert pdf2['scores'].tolist() == [100, 200, 300, 400], "df['col'] = list must use list values"
    
    # Test 3: df['new_col'] = Polars expression
    df3 = npd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, 20, 30, 40]
    })
    df3['doubled'] = pl.col("value") * 2
    assert "doubled" in df3.columns, "df['col'] = Polars expression must add column"
    assert df3.shape == (4, 3), "df['col'] = expression must add column"
    pdf3 = df3.to_pandas()
    assert pdf3['doubled'].tolist() == [20, 40, 60, 80], "df['col'] = expression must calculate correctly"
    
    # Test 4: df['existing_col'] = new value (modify existing column)
    df4 = npd.DataFrame({
        'id': [1, 2, 3, 4],
        'value': [10, 20, 30, 40]
    })
    df4['value'] = pl.col("value") * 10
    pdf4 = df4.to_pandas()
    assert pdf4['value'].tolist() == [100, 200, 300, 400], "df['col'] = value must modify existing column"
    
    # Test 5: df['new_col'] = pandas Series
    df5 = npd.DataFrame({'a': [1, 2, 3]})
    series = pd.Series([10, 20, 30])
    df5['b'] = series
    assert "b" in df5.columns, "df['col'] = pandas Series must add column"
    pdf5 = df5.to_pandas()
    assert pdf5['b'].tolist() == [10, 20, 30], "df['col'] = pandas Series must use Series values"
    
    # Test 6: df['new_col'] = string scalar
    df6 = npd.DataFrame({'id': [1, 2, 3]})
    df6['status'] = "active"
    assert "status" in df6.columns, "df['col'] = string scalar must add column"
    pdf6 = df6.to_pandas()
    assert pdf6['status'].tolist() == ["active", "active", "active"], "df['col'] = string must fill all rows"
    
    # Test 7: df['new_col'] = expression using multiple columns
    df7 = npd.DataFrame({
        'a': [1, 2, 3],
        'b': [10, 20, 30]
    })
    df7['sum'] = pl.col("a") + pl.col("b")
    assert "sum" in df7.columns, "df['col'] = multi-column expression must add column"
    pdf7 = df7.to_pandas()
    assert pdf7['sum'].tolist() == [11, 22, 33], "df['col'] = multi-column expression must calculate correctly"
    
    print("OK Test df['col'] = value (setitem) OK")