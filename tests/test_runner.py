"""Runner pour exécuter tous les tests"""
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_io import (
    test_read_csv,
    test_read_parquet,
    test_read_parquet_enhanced,
    test_read_excel,
    test_read_json,
    test_read_json_enhanced,
    test_read_csv_lazy,
    test_read_csv_lazy_enhanced,
    test_write_csv_enhanced,
    test_read_parquet_lazy,
    test_to_csv,
    test_to_parquet,
    test_to_json,
    test_to_excel,
    test_read_json_lazy_new,
    test_read_excel_lazy_new,
    test_read_excel_lazy_enhanced,
    test_dtype_casting,
)

from test_dataframe import (
    test_dataframe_creation,
    test_query_method,
    test_loc_iloc,
    test_loc_mask,
    test_loc_mask_dataframe,
    test_loc_direct_comparison,
    test_loc_mask_col,
    test_direct_mask,
    test_dataframe_slicing,
    test_pandas_fallback_describe,
    test_pandas_fallback_module_level,
    test_sort_values_and_rename_and_drop_and_astype_and_fillna,
    test_drop_with_axis,
    test_drop_duplicates_and_value_counts_and_reset_index,
    test_merge_and_concat_and_isna_notna,
    test_concat_content,
    test_setitem_add_column,
)

from test_groupby import (
    test_groupby,
    test_groupby_lazy,
    test_groupby_agg_lambda,
)

if __name__ == "__main__":
    # Tests IO
    test_read_csv()
    test_read_parquet()
    test_read_parquet_enhanced()
    test_read_excel()
    test_read_json()
    test_read_json_enhanced()
    test_read_csv_lazy()
    test_read_csv_lazy_enhanced()
    test_write_csv_enhanced()
    test_read_parquet_lazy()
    test_to_csv()
    test_to_parquet()
    test_to_json()
    test_to_excel()
    test_read_json_lazy_new()
    test_read_excel_lazy_new()
    test_read_excel_lazy_enhanced()
    test_dtype_casting()
    
    # Tests DataFrame
    test_dataframe_creation()
    test_query_method()
    test_loc_iloc()
    test_loc_mask()
    test_loc_mask_dataframe()
    test_loc_direct_comparison()
    test_loc_mask_col()
    test_direct_mask()
    test_dataframe_slicing()
    test_pandas_fallback_describe()
    test_pandas_fallback_module_level()
    test_sort_values_and_rename_and_drop_and_astype_and_fillna()
    test_drop_with_axis()
    test_drop_duplicates_and_value_counts_and_reset_index()
    test_merge_and_concat_and_isna_notna()
    test_concat_content()
    test_setitem_add_column()
    
    # Tests groupby
    test_groupby()
    test_groupby_lazy()
    test_groupby_agg_lambda()
    
    print("\nOK Tous les tests sont passes avec succes !")

