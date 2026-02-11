
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
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
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val - min_val == 0:
        return data
    
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: input DataFrame
        numeric_columns: list of numeric column names
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Args:
        df: pandas DataFrame
        column: column name
    
    Returns:
        Dictionary with statistics
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

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics for column 'A':")
    print(calculate_statistics(sample_data, 'A'))
    
    cleaned_data = clean_dataset(sample_data, ['A', 'B'])
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics for column 'A':")
    print(calculate_statistics(cleaned_data, 'A'))import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Remove rows where all values are NaN
        df.dropna(how='all', inplace=True)

        # Reset index after cleaning
        df.reset_index(drop=True, inplace=True)

        print(f"Cleaned data shape: {df.shape}")
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")

        return df

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    cleaned_df = clean_data(input_csv, output_csv)