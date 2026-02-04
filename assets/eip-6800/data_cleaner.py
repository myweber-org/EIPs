
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (list or array-like): The dataset containing the column to clean.
    column (int or str): Index or name of the column to process.
    
    Returns:
    tuple: (cleaned_data, outlier_indices)
    """
    if isinstance(data, list):
        data_array = np.array(data)
    else:
        data_array = np.asarray(data)
    
    if isinstance(column, str):
        raise ValueError("Column names not supported with array input. Use integer index.")
    
    column_data = data_array[:, column].astype(float)
    
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (column_data < lower_bound) | (column_data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    
    cleaned_data = data_array[~outlier_mask]
    
    return cleaned_data, outlier_indices

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    data (array-like): The dataset.
    column (int): Index of the column.
    
    Returns:
    dict: Dictionary containing statistics.
    """
    column_data = data[:, column].astype(float)
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'q1': np.percentile(column_data, 25),
        'q3': np.percentile(column_data, 75)
    }
    
    return stats

if __name__ == "__main__":
    sample_data = np.array([
        [1, 150.5],
        [2, 165.2],
        [3, 172.8],
        [4, 158.1],
        [5, 210.0],
        [6, 155.3],
        [7, 300.0],
        [8, 162.7],
        [9, 168.9],
        [10, 152.4]
    ])
    
    print("Original data shape:", sample_data.shape)
    print("Original data:")
    print(sample_data)
    
    cleaned, outliers = remove_outliers_iqr(sample_data, 1)
    
    print("\nOutlier indices:", outliers)
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned data:")
    print(cleaned)
    
    stats = calculate_basic_stats(sample_data, 1)
    print("\nStatistics for column 1:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import pandas as pd
import numpy as np

def clean_csv_data(filepath, output_path=None):
    """
    Load a CSV file, clean missing values, and convert data types.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        
        # Save cleaned data
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        
        # Print summary
        print(f"Data cleaning completed:")
        print(f"  - Rows processed: {initial_rows}")
        print(f"  - Duplicates removed: {duplicates_removed}")
        print(f"  - Missing values handled")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Warning: DataFrame still contains {null_counts.sum()} null values.")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  - {col}: {count} nulls")
    
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data validation passed.")
        else:
            print("Data validation failed.")