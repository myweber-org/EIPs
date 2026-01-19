
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_na_threshold=0.5):
    """
    Clean a dataset by handling missing values, removing duplicates,
    and optionally renaming columns.
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Rename columns if mapping is provided
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # Calculate missing value percentage per column
    missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
    
    # Drop columns with too many missing values
    columns_to_drop = missing_percent[missing_percent > drop_na_threshold * 100].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Fill remaining missing values with appropriate defaults
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('Unknown')
        elif np.issubdtype(df_clean[col].dtype, np.number):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Generate cleaning report
    report = {
        'original_rows': len(df),
        'cleaned_rows': len(df_clean),
        'duplicates_removed': duplicates_removed,
        'columns_dropped': list(columns_to_drop),
        'missing_values_filled': df.isnull().sum().sum() - df_clean.isnull().sum().sum()
    }
    
    return df_clean, report

def validate_dataframe(df, required_columns=None, unique_columns=None):
    """
    Validate dataframe structure and constraints.
    """
    validation_results = {}
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['has_all_required_columns'] = len(missing_columns) == 0
    
    # Check uniqueness constraints
    if unique_columns:
        for col in unique_columns:
            if col in df.columns:
                is_unique = df[col].is_unique
                validation_results[f'{col}_is_unique'] = is_unique
                if not is_unique:
                    validation_results[f'{col}_duplicate_count'] = len(df[col]) - len(df[col].drop_duplicates())
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    negative_counts = {}
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            negative_counts[col] = negative_count
    validation_results['negative_value_counts'] = negative_counts
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5],
        'department': ['HR', 'IT', 'IT', 'Finance', None, 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df, report = clean_dataset(df, drop_na_threshold=0.3)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Validate the cleaned data
    validation = validate_dataframe(
        cleaned_df,
        required_columns=['id', 'name', 'age'],
        unique_columns=['id']
    )
    
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
import numpy as np
import pandas as pd

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
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    # Add some outliers
    data['value'][95] = 500
    data['value'][96] = -200
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:", calculate_statistics(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = main()