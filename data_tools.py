from crewai.tools import tool
import pandas as pd
import numpy as np


@tool("Load customer data from CSV file")
def data_load_tool(file_path: str) -> str:
    """Loads customer data from CSV file and provides basic information.
    
    Args:
        file_path: Path to the CSV file to load
    """
    try:
        df = pd.read_csv(file_path)
        
        summary = f"""
Data loaded successfully!

Shape: {df.shape[0]} rows, {df.shape[1]} columns

Columns: {', '.join(df.columns.tolist())}

First few rows:
{df.head(3).to_string()}

Data types:
{df.dtypes.to_string()}

Missing values:
{df.isnull().sum().to_string()}

Basic statistics:
{df.describe().to_string()}
"""
        return summary
    except Exception as e:
        return f"Error loading data: {str(e)}"


@tool("Clean customer data by handling missing values")
def data_clean_tool(file_path: str) -> str:
    """Cleans the dataset by handling missing values and data quality issues.
    
    Args:
        file_path: Path to the CSV file to clean
    """
    try:
        df = pd.read_csv(file_path)
        initial_shape = df.shape
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric with median
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Save cleaned data
        cleaned_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_path, index=False)
        
        return f"""
Data cleaning completed!

Initial shape: {initial_shape}
Final shape: {df.shape}
Rows removed: {initial_shape[0] - df.shape[0]}

Missing values after cleaning: {df.isnull().sum().sum()} total missing values

Cleaned data saved to: {cleaned_path}
"""
    except Exception as e:
        return f"Error cleaning data: {str(e)}"