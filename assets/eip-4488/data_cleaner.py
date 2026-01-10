
import pandas as pd

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_check (list, optional): Specific columns to check for duplicates.
            If None, checks all columns. Defaults to None.
        fill_missing (bool or dict, optional): Strategy to fill missing values.
            If True, fills with column mean for numeric and mode for categorical.
            If dict, uses provided fill values. Defaults to True.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check:
        cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check)
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        else:
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype in ['int64', 'float64']:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown')
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): The dataset
    column (int): Index of the column to clean
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    column_data = data[:, column]
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy.ndarray): The dataset
    column (int): Index of the column to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    cleaned_data = remove_outliers_iqr(data, column)
    column_data = cleaned_data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'count': len(column_data)
    }
    
    return stats

def validate_data(data):
    """
    Validate input data for cleaning operations.
    
    Parameters:
    data: Input data to validate
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    if not isinstance(data, np.ndarray):
        return False
    
    if data.size == 0:
        return False
    
    if len(data.shape) != 2:
        return False
    
    return True

def example_usage():
    """
    Demonstrate how to use the data cleaning functions.
    """
    np.random.seed(42)
    
    sample_data = np.random.randn(100, 3)
    sample_data[0, 0] = 10  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    
    if validate_data(sample_data):
        cleaned_data = remove_outliers_iqr(sample_data, 0)
        print("Cleaned data shape:", cleaned_data.shape)
        
        stats = calculate_statistics(sample_data, 0)
        print("Statistics for column 0:", stats)
    else:
        print("Invalid data provided")

if __name__ == "__main__":
    example_usage()