
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if cleaned_df.isnull().sum().sum() > 0:
        print(f"Found {cleaned_df.isnull().sum().sum()} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with missing values")
        elif fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            print("Filled missing numeric values with column means")
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print("Filled missing numeric values with column medians")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().any():
                    if cleaned_df[col].dtype == 'object':
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
            print("Filled missing categorical values with column modes")
    
    print(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame including basic statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode()[0] if not df[col].mode().empty else None
        }
    
    return summary
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df, date_column='date', id_column='id'):
    """
    Clean dataframe by removing duplicates and standardizing date formats.
    
    Args:
        df: pandas DataFrame to clean
        date_column: name of date column to standardize
        id_column: name of ID column for duplicate checking
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Remove duplicate rows based on ID column
    if id_column in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=[id_column], keep='first')
    
    # Standardize date format if date column exists
    if date_column in cleaned_df.columns:
        try:
            # Try to parse dates in multiple formats
            cleaned_df[date_column] = pd.to_datetime(
                cleaned_df[date_column], 
                errors='coerce',
                format='mixed'
            )
            
            # Format dates to YYYY-MM-DD
            cleaned_df[date_column] = cleaned_df[date_column].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error standardizing dates: {e}")
    
    # Fill missing numeric values with column mean
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill missing string values with 'Unknown'
    string_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in string_cols:
        if col != date_column:  # Skip date column
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_rows': 0,
        'null_percentage': {}
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['missing_columns'] = missing
            validation_results['is_valid'] = False
    
    # Count empty rows
    validation_results['empty_rows'] = df.isnull().all(axis=1).sum()
    
    # Calculate null percentage for each column
    for column in df.columns:
        null_count = df[column].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        validation_results['null_percentage'][column] = round(null_pct, 2)
    
    return validation_results

def export_cleaned_data(df, output_path, format='csv'):
    """
    Export cleaned dataframe to file.
    
    Args:
        df: pandas DataFrame to export
        output_path: path for output file
        format: output format ('csv', 'excel', 'json')
    
    Returns:
        Boolean indicating success
    """
    try:
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data successfully exported to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'date': ['2023-01-15', '15/02/2023', '2023-03-20', '20-04-2023', None, '2023-05-10'],
        'value': [100, 200, 200, 300, None, 500],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, date_column='date', id_column='id')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate the data
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'date', 'value'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")