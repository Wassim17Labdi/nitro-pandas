"""
Tests for groupby operations.

This module contains comprehensive tests for nitro-pandas groupby functionality,
including aggregation methods, column selection, and lazy groupby operations.
"""

import warnings
warnings.filterwarnings('ignore')
import nitro_pandas as npd
import pandas as pd
import polars as pl
try:
    from .helpers import create_sample_csv
except ImportError:
    from helpers import create_sample_csv


def test_groupby():
    """
    Test groupby operations with various aggregation methods.
    
    This test verifies that groupby works correctly with:
    - Basic aggregation methods (mean, sum, min, max, count)
    - Multi-column grouping
    - Column selection with groupby()["col"]
    - Dictionary-based aggregation with agg()
    """
    csv_path = create_sample_csv()
    try:
        df = npd.read_csv(csv_path)
        
        # Test 1: groupby().mean()
        result_mean = df.groupby("name").mean()
        assert isinstance(result_mean, npd.DataFrame), "groupby().mean() must return a DataFrame"
        assert "name" in result_mean.columns, "groupby().mean() must keep grouping column"
        assert result_mean.shape[0] > 0, "groupby().mean() must return results"
        
        # Test 2: groupby().sum()
        result_sum = df.groupby("name").sum()
        assert isinstance(result_sum, npd.DataFrame), "groupby().sum() must return a DataFrame"
        assert result_sum.shape[0] > 0, "groupby().sum() must return results"
        
        # Test 3: groupby().min()
        result_min = df.groupby("name").min()
        assert isinstance(result_min, npd.DataFrame), "groupby().min() must return a DataFrame"
        assert result_min.shape[0] > 0, "groupby().min() must return results"
        
        # Test 4: groupby().max()
        result_max = df.groupby("name").max()
        assert isinstance(result_max, npd.DataFrame), "groupby().max() must return a DataFrame"
        assert result_max.shape[0] > 0, "groupby().max() must return results"
        
        # Test 5: groupby().count()
        result_count = df.groupby("name").count()
        assert isinstance(result_count, npd.DataFrame), "groupby().count() must return a DataFrame"
        assert result_count.shape[0] > 0, "groupby().count() must return results"
        # Verify count returns values >= 1
        count_values = result_count.to_pandas()["id"].tolist() if "id" in result_count.columns else []
        assert all(c >= 1 for c in count_values), "groupby().count() must return counts >= 1"
        
        # Test 6: Multi-column groupby
        # Create DataFrame directly with dict (pandas-like syntax)
        df_multi = npd.DataFrame({
            "cat": ["A", "A", "B", "B", "A"],
            "subcat": ["X", "Y", "X", "Y", "X"],
            "value": [10, 20, 30, 40, 15]
        })
        result_multi = df_multi.groupby(["cat", "subcat"]).mean()
        assert isinstance(result_multi, npd.DataFrame), "Multi-column groupby() must work"
        assert "cat" in result_multi.columns and "subcat" in result_multi.columns, "groupby() must keep all grouping columns"
        
        # Test 7: Multi-column groupby with column selection
        # df.groupby(["ville", "sexe"])["revenu"].sum()
        df_multi_col = npd.DataFrame({
            "ville": ["Paris", "Paris", "Lyon", "Lyon", "Paris"],
            "sexe": ["M", "F", "M", "F", "M"],
            "revenu": [1000, 2000, 1500, 1800, 1200]
        })
        result_multi_col_sum = df_multi_col.groupby(["ville", "sexe"])["revenu"].sum()
        assert isinstance(result_multi_col_sum, npd.DataFrame), "Multi-column groupby() with selection must return DataFrame"
        assert "ville" in result_multi_col_sum.columns, "groupby() must keep all grouping columns"
        assert "sexe" in result_multi_col_sum.columns, "groupby() must keep all grouping columns"
        assert "revenu" in result_multi_col_sum.columns, "groupby() must have aggregated column"
        assert result_multi_col_sum.shape[0] > 0, "groupby() must return results"
        
        # Test 8: Multi-column groupby with column selection and mean
        result_multi_col_mean = df_multi_col.groupby(["ville", "sexe"])["revenu"].mean()
        assert isinstance(result_multi_col_mean, npd.DataFrame), "Multi-column groupby() with selection must return DataFrame"
        assert "ville" in result_multi_col_mean.columns and "sexe" in result_multi_col_mean.columns, "groupby() must keep all grouping columns"
        assert "revenu" in result_multi_col_mean.columns, "groupby() must have aggregated column"
        
        # Test 9: groupby().agg() with pandas-like dictionary (without using pl.col())
        # Create DataFrame with multiple numeric columns
        df_agg = npd.DataFrame({
            "name": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
            "value2": [100, 200, 300, 400]
        })
        result_agg_dict = df_agg.groupby("name").agg({
            "value": "mean",
            "value2": "sum"
        })
        assert isinstance(result_agg_dict, npd.DataFrame), "groupby().agg() with dict must return DataFrame"
        assert "name" in result_agg_dict.columns, "groupby().agg() must keep grouping column"
        assert "value" in result_agg_dict.columns, "groupby().agg() must create specified column"
        assert "value2" in result_agg_dict.columns, "groupby().agg() must create specified column"
        assert result_agg_dict.shape[0] > 0, "groupby().agg() must return results"
        
        # Test 10: groupby().agg() with single aggregation
        result_agg_single = df_agg.groupby("name").agg({
            "value": "mean"
        })
        assert isinstance(result_agg_single, npd.DataFrame), "groupby().agg() with single aggregation must work"
        assert "value" in result_agg_single.columns, "groupby().agg() must create specified column"
        
        # Test 11: groupby()["colonne"].mean() - pandas-like syntax, Polars backend, returns DataFrame
        result_col_mean = df.groupby("name")["value"].mean()
        assert isinstance(result_col_mean, npd.DataFrame), "groupby()['colonne'].mean() must return DataFrame nitro-pandas"
        assert result_col_mean.shape[0] > 0, "groupby()['colonne'].mean() must return results"
        assert "name" in result_col_mean.columns, "groupby()['colonne'].mean() must keep grouping column"
        assert "value" in result_col_mean.columns, "groupby()['colonne'].mean() must have aggregated column"
        
        # Test 12: groupby()["colonne"].sum()
        result_col_sum = df.groupby("name")["value"].sum()
        assert isinstance(result_col_sum, npd.DataFrame), "groupby()['colonne'].sum() must return DataFrame nitro-pandas"
        assert result_col_sum.shape[0] > 0, "groupby()['colonne'].sum() must return results"
        assert "name" in result_col_sum.columns and "value" in result_col_sum.columns
        
        # Test 13: groupby()["colonne"].min()
        result_col_min = df.groupby("name")["value"].min()
        assert isinstance(result_col_min, npd.DataFrame), "groupby()['colonne'].min() must return DataFrame nitro-pandas"
        assert "name" in result_col_min.columns and "value" in result_col_min.columns
        
        # Test 14: groupby()["colonne"].max()
        result_col_max = df.groupby("name")["value"].max()
        assert isinstance(result_col_max, npd.DataFrame), "groupby()['colonne'].max() must return DataFrame nitro-pandas"
        assert "name" in result_col_max.columns and "value" in result_col_max.columns
        
        # Test 15: groupby()["colonne"].count()
        result_col_count = df.groupby("name")["value"].count()
        assert isinstance(result_col_count, npd.DataFrame), "groupby()['colonne'].count() must return DataFrame nitro-pandas"
        assert "name" in result_col_count.columns and "value" in result_col_count.columns
        
        print("OK Test groupby OK")
    finally:
        import os
        os.unlink(csv_path)


def test_groupby_lazy():
    """
    Test lazy groupby operations with LazyFrame.
    
    This test verifies that groupby works correctly with LazyFrame,
    including aggregation methods and column selection.
    """
    csv_path = create_sample_csv()
    try:
        lf = npd.read_csv_lazy(csv_path)
        
        # Test 1: LazyFrame groupby().mean()
        result_mean = lf.groupby("name").mean()
        df_mean = result_mean.collect()
        assert isinstance(df_mean, npd.DataFrame), "LazyFrame groupby().mean() must return DataFrame after collect()"
        assert "name" in df_mean.columns, "LazyFrame groupby().mean() must keep grouping column"
        assert df_mean.shape[0] > 0, "LazyFrame groupby().mean() must return results"
        
        # Test 2: LazyFrame groupby().sum()
        result_sum = lf.groupby("name").sum()
        df_sum = result_sum.collect()
        assert isinstance(df_sum, npd.DataFrame), "LazyFrame groupby().sum() must return DataFrame"
        assert df_sum.shape[0] > 0, "LazyFrame groupby().sum() must return results"
        
        # Test 3: LazyFrame groupby().count()
        result_count = lf.groupby("name").count()
        df_count = result_count.collect()
        assert isinstance(df_count, npd.DataFrame), "LazyFrame groupby().count() must return DataFrame"
        assert df_count.shape[0] > 0, "LazyFrame groupby().count() must return results"
        
        # Test 4: LazyFrame groupby() with column selection
        result_col = lf.groupby("name")["value"].mean()
        df_col = result_col.collect()
        assert isinstance(df_col, npd.DataFrame), "LazyFrame groupby() with selection must return DataFrame"
        assert "name" in df_col.columns and "value" in df_col.columns, "LazyFrame groupby() must keep columns"
        
        print("OK Test groupby lazy OK")
    finally:
        import os
        os.unlink(csv_path)


def test_groupby_agg_lambda():
    """
    Test groupby with agg() using lambda expressions.
    
    This test verifies that groupby().agg() correctly handles lambda functions
    via map_elements(), allowing custom Python functions to be applied.
    """
    # Create test DataFrame with numeric data
    df = npd.DataFrame({
        'category': ['A', 'A', 'B', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40, 15, 50],
        'score': [100, 200, 300, 400, 150, 500]
    })
    
    # Test 1: groupby().agg() with lambda that calculates mean manually: x.sum() / len(x)
    result_mean_lambda = df.groupby("category").agg({
        "value": lambda x: x.sum() / len(x)
    })
    assert isinstance(result_mean_lambda, npd.DataFrame), "groupby().agg() with lambda must return DataFrame"
    assert "category" in result_mean_lambda.columns, "groupby().agg() must keep grouping column"
    assert "value" in result_mean_lambda.columns, "groupby().agg() must create aggregated column"
    
    # Verify content: lambda calculates mean manually
    # For category 'A': values are [10, 20, 15] -> sum = 45, len = 3 -> mean = 15.0
    # For category 'B': values are [30, 40, 50] -> sum = 120, len = 3 -> mean = 40.0
    pdf_result = result_mean_lambda.to_pandas()
    assert pdf_result.shape[0] == 2, "groupby().agg() with lambda must return one row per group"
    
    # Verify the calculated means - exact values expected
    result_dict = pdf_result.set_index('category')['value'].to_dict()
    assert result_dict['A'] == 15.0, f"Category A mean should be exactly 15.0, got {result_dict['A']}"
    assert result_dict['B'] == 40.0, f"Category B mean should be exactly 40.0, got {result_dict['B']}"
    
    # Alternative: verify using direct value access
    category_a_row = pdf_result[pdf_result['category'] == 'A']
    category_b_row = pdf_result[pdf_result['category'] == 'B']
    assert category_a_row['value'].iloc[0] == 15.0, f"Category A value should be 15.0, got {category_a_row['value'].iloc[0]}"
    assert category_b_row['value'].iloc[0] == 40.0, f"Category B value should be 40.0, got {category_b_row['value'].iloc[0]}"
    
    # Test 2: groupby().agg() with lambda x.sum() / len(x) on score column
    result_score_mean = df.groupby("category").agg({
        "score": lambda x: x.sum() / len(x)
    })
    assert isinstance(result_score_mean, npd.DataFrame), "groupby().agg() with lambda on score must return DataFrame"
    assert "score" in result_score_mean.columns, "groupby().agg() must create score column"
    
    # Verify score means - exact values expected
    # For category 'A': scores are [100, 200, 150] -> sum = 450, len = 3 -> mean = 150.0
    # For category 'B': scores are [300, 400, 500] -> sum = 1200, len = 3 -> mean = 400.0
    pdf_score = result_score_mean.to_pandas()
    score_dict = pdf_score.set_index('category')['score'].to_dict()
    assert score_dict['A'] == 150.0, f"Category A score mean should be exactly 150.0, got {score_dict['A']}"
    assert score_dict['B'] == 400.0, f"Category B score mean should be exactly 400.0, got {score_dict['B']}"
    
    # Verify using direct row access
    score_a_row = pdf_score[pdf_score['category'] == 'A']
    score_b_row = pdf_score[pdf_score['category'] == 'B']
    assert score_a_row['score'].iloc[0] == 150.0, f"Category A score should be 150.0, got {score_a_row['score'].iloc[0]}"
    assert score_b_row['score'].iloc[0] == 400.0, f"Category B score should be 400.0, got {score_b_row['score'].iloc[0]}"
    
    # Test 3: groupby().agg() with multiple lambdas using x.sum() / len(x) on different columns
    result_multi = df.groupby("category").agg({
        "value": lambda x: x.sum() / len(x),
        "score": lambda x: x.sum() / len(x)
    })
    assert isinstance(result_multi, npd.DataFrame), "groupby().agg() with multiple lambdas must return DataFrame"
    assert "value" in result_multi.columns, "groupby().agg() must have value column"
    assert "score" in result_multi.columns, "groupby().agg() must have score column"
    assert result_multi.shape[0] == 2, "groupby().agg() must return one row per group"
    
    # Verify exact values for both columns
    pdf_multi = result_multi.to_pandas()
    multi_a = pdf_multi[pdf_multi['category'] == 'A'].iloc[0]
    multi_b = pdf_multi[pdf_multi['category'] == 'B'].iloc[0]
    assert multi_a['value'] == 15.0, f"Category A value should be 15.0, got {multi_a['value']}"
    assert multi_a['score'] == 150.0, f"Category A score should be 150.0, got {multi_a['score']}"
    assert multi_b['value'] == 40.0, f"Category B value should be 40.0, got {multi_b['value']}"
    assert multi_b['score'] == 400.0, f"Category B score should be 400.0, got {multi_b['score']}"
    
    # Test 4: groupby().agg() with lambda x.sum() / len(x) and string function together
    result_mixed = df.groupby("category").agg({
        "value": lambda x: x.sum() / len(x),  # Lambda calculating mean
        "score": "mean"  # String function (should give same result)
    })
    assert isinstance(result_mixed, npd.DataFrame), "groupby().agg() with mixed lambda and string must return DataFrame"
    assert "value" in result_mixed.columns, "groupby().agg() must have value column"
    assert "score" in result_mixed.columns, "groupby().agg() must have score column"
    
    # Verify that lambda mean and string mean give same results for score
    pdf_mixed = result_mixed.to_pandas()
    mixed_a = pdf_mixed[pdf_mixed['category'] == 'A'].iloc[0]
    mixed_b = pdf_mixed[pdf_mixed['category'] == 'B'].iloc[0]
    
    # Verify lambda-calculated value column
    assert mixed_a['value'] == 15.0, f"Category A value (lambda) should be 15.0, got {mixed_a['value']}"
    assert mixed_b['value'] == 40.0, f"Category B value (lambda) should be 40.0, got {mixed_b['value']}"
    
    # Verify string function mean on score column
    assert mixed_a['score'] == 150.0, f"Category A score (mean) should be 150.0, got {mixed_a['score']}"
    assert mixed_b['score'] == 400.0, f"Category B score (mean) should be 400.0, got {mixed_b['score']}"
    
    print("OK Test groupby agg with lambda OK")