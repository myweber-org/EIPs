
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def main():
    data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10]}
    df = pd.DataFrame(data)
    print("Original data:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['values'])
    print("\nCleaned data:")
    print(cleaned_df)

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def normalize_column(df, column):
    """Normalize column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    """Apply outlier removal and normalization to numeric columns."""
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers(df, col)
            df = normalize_column(df, col)
    return df

if __name__ == "__main__":
    data = load_data('sample_data.csv')
    numeric_cols = ['age', 'income', 'score']
    cleaned_data = clean_data(data, numeric_cols)
    cleaned_data.to_csv('cleaned_data.csv', index=False)
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistics
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def main():
    # Example usage
    data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 14, 13, 12, 11, 14, 13, 12, 14, 13]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    
    # Calculate statistics before cleaning
    original_stats = calculate_statistics(df, 'values')
    print("\nOriginal statistics:")
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Remove outliers
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nCleaned shape: {cleaned_df.shape}")
    
    # Calculate statistics after cleaning
    cleaned_stats = calculate_statistics(cleaned_df, 'values')
    print("\nCleaned statistics:")
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    # Show removed outliers
    outliers = df[~df.index.isin(cleaned_df.index)]
    print(f"\nRemoved outliers ({len(outliers)}):")
    print(outliers)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV
    missing_strategy (str): Strategy for handling missing values
                           ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    # Read the CSV file
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif missing_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif missing_strategy == 'zero':
        df = df.fillna(0)
    elif missing_strategy == 'drop':
        df = df.dropna()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save cleaned data if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"  - Initial rows: {initial_rows}")
    print(f"  - Duplicates removed: {duplicates_removed}")
    print(f"  - Final rows: {len(df)}")
    print(f"  - Missing values handled using: {missing_strategy} strategy")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if dataframe is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('Dataframe is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for all-NaN columns
    nan_columns = df.columns[df.isna().all()].tolist()
    if nan_columns:
        validation_results['warnings'].append(f'Columns with all NaN values: {nan_columns}')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, np.nan, 15.2, np.nan, 20.1, 20.1],
        'category': ['A', 'B', 'A', 'C', 'B', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    test_input = 'test_input.csv'
    df.to_csv(test_input, index=False)
    
    try:
        cleaned_df = clean_csv_data(test_input, 'test_output.csv', 'mean')
        validation = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Validation results: {validation}")
    finally:
        # Clean up test files
        for file in [test_input, 'test_output.csv']:
            if Path(file).exists():
                Path(file).unlink()import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data
    
    data[column] = (data[column] - min_val) / (max_val - min_val)
    return data

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data
    
    data[column] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(data, numeric_columns, outlier_factor=1.5, normalize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_factor: IQR factor for outlier removal
        normalize: whether to apply min-max normalization
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_factor)
            if normalize:
                cleaned_data = normalize_minmax(cleaned_data, col)
    
    return cleaned_data

def get_summary_statistics(data, numeric_columns):
    """
    Calculate summary statistics for numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        DataFrame with summary statistics
    """
    summary = {}
    
    for col in numeric_columns:
        if col in data.columns:
            summary[col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'count': data[col].count(),
                'missing': data[col].isnull().sum()
            }
    
    return pd.DataFrame(summary).T

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("\nOriginal summary statistics:")
    print(get_summary_statistics(sample_data, ['feature1', 'feature2', 'feature3']))
    
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2', 'feature3'])
    
    print("\nCleaned data shape:", cleaned.shape)
    print("\nCleaned summary statistics:")
    print(get_summary_statistics(cleaned, ['feature1', 'feature2', 'feature3']))
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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
    
    return filtered_df.copy()

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

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.array([500, 600, -100])
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 98)
    }
    
    df = pd.DataFrame(data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['values'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    
    stats = calculate_summary_statistics(cleaned_df, 'values')
    print("\nSummary statistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")