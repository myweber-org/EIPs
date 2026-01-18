
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def example_usage():
    """
    Example demonstrating the usage of data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_summary_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_summary_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing columns
    with excessive missing data.
    
    Parameters:
    file_path (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Threshold for dropping columns (0.0 to 1.0)
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")
    
    original_shape = df.shape
    
    # Calculate missing percentage per column
    missing_percent = df.isnull().sum() / len(df)
    
    # Drop columns with missing data above threshold
    columns_to_drop = missing_percent[missing_percent > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Fill remaining missing values based on method
    if fill_method == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif fill_method == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif fill_method == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif fill_method == 'zero':
        df = df.fillna(0)
    else:
        raise ValueError(f"Unsupported fill method: {fill_method}")
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    cleaned_shape = df.shape
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {cleaned_shape}")
    print(f"Columns dropped: {len(columns_to_drop)}")
    print(f"Missing values filled: {df.isnull().sum().sum()}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values
    if df.select_dtypes(include=[np.number]).size > 0:
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            validation_results['warnings'].append(f'Found {inf_count} infinite values')
    
    # Check data types
    dtype_info = df.dtypes.to_dict()
    validation_results['dtypes'] = dtype_info
    
    return validation_results

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Parameters:
    df (pandas.DataFrame): DataFrame to save
    output_path (str): Output file path
    format (str): Output format ('csv', 'excel', 'json')
    """
    
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # This would be used for testing the functions
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 1, 1, 1, 1],
        'D': [10, 20, 30, 40, 50]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_method='mean', drop_threshold=0.3)
    
    validation = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'D'])
    print(f"Validation results: {validation}")
    
    save_cleaned_data(cleaned_df, 'cleaned_data.csv')
    
    # Clean up test file
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')