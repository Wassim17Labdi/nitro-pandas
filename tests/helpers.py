"""
Test helper functions for creating sample data files.

This module provides utility functions to create temporary test files
in various formats (CSV, Parquet, Excel, JSON) for use in test suites.
"""

import tempfile
import os
import json
import pandas as pd


def create_sample_csv():
    """
    Create a temporary CSV file for testing.
    
    Returns:
        str: Path to the temporary CSV file
        
    Note:
        The file must be manually deleted after use (see test cleanup).
    """
    content = """id,name,value,date
1,John,10.5,2024-01-01
2,Jane,20.3,2024-01-02
3,Bob,"30,000.75",2024-01-03
4,Alice,,2024-01-04"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(content)
        return f.name


def create_sample_parquet():
    """
    Create a temporary Parquet file for testing.
    
    Returns:
        str: Path to the temporary Parquet file
        
    Note:
        The file must be manually deleted after use (see test cleanup).
    """
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['John', 'Jane', 'Bob', 'Alice'],
        'value': [10.5, 20.3, 30.75, None],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as f:
        df.to_parquet(f.name)
        return f.name


def create_sample_excel():
    """
    Create a temporary Excel file for testing.
    
    Returns:
        str: Path to the temporary Excel file
        
    Note:
        The file must be manually deleted after use (see test cleanup).
    """
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'name': ['John', 'Jane', 'Bob', 'Alice'],
        'value': [10.5, 20.3, 30.75, None],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as f:
        df.to_excel(f.name, index=False)
        return f.name


def create_sample_json():
    """
    Create a temporary JSON file for testing.
    
    Returns:
        str: Path to the temporary JSON file
        
    Note:
        The file must be manually deleted after use (see test cleanup).
    """
    data = [
        {"id": 1, "name": "John", "value": 10.5, "date": "2024-01-01"},
        {"id": 2, "name": "Jane", "value": 20.3, "date": "2024-01-02"},
        {"id": 3, "name": "Bob", "value": 30.75, "date": "2024-01-03"},
        {"id": 4, "name": "Alice", "value": None, "date": "2024-01-04"}
    ]
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(data, f)
        return f.name
