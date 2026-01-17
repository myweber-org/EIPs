def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def clean_dataset(df, numeric_columns):
    df_cleaned = df.dropna(subset=numeric_columns)
    df_no_outliers = remove_outliers_iqr(df_cleaned, numeric_columns)
    df_normalized = normalize_minmax(df_no_outliers, numeric_columns)
    return df_normalized

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1000, 200)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    cleaned_data = clean_dataset(sample_df, numeric_cols)
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print("\nCleaned data summary:")
    print(cleaned_data.describe())
import pandas as pd

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = cleaned_df.shape[0]
    cleaned_df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - cleaned_df.shape[0]
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns
    
    missing_info = {}
    for column in columns_to_check:
        if column in cleaned_df.columns:
            missing_count = cleaned_df[column].isnull().sum()
            if missing_count > 0:
                # For numeric columns, fill with median
                if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    median_value = cleaned_df[column].median()
                    cleaned_df[column].fillna(median_value, inplace=True)
                    missing_info[column] = {
                        'missing_count': missing_count,
                        'method': 'median',
                        'value': median_value
                    }
                # For categorical columns, fill with mode
                else:
                    mode_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                    cleaned_df[column].fillna(mode_value, inplace=True)
                    missing_info[column] = {
                        'missing_count': missing_count,
                        'method': 'mode',
                        'value': mode_value
                    }
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"  - Removed {removed_duplicates} duplicate rows")
    print(f"  - Original shape: {df.shape}")
    print(f"  - Cleaned shape: {cleaned_df.shape}")
    
    if missing_info:
        print(f"  - Missing values handled in {len(missing_info)} columns:")
        for col, info in missing_info.items():
            print(f"    * {col}: {info['missing_count']} values filled with {info['method']} ({info['value']})")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

# Example usage function
def example_usage():
    """
    Demonstrate how to use the data cleaning functions.
    """
    # Create sample data
    data = {
        'id': [1, 2, 3, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', None, 'Eve'],
        'age': [25, 30, None, 35, 28, 32],
        'score': [85.5, 92.0, 78.5, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df)
    
    print("\n" + "="*50 + "\n")
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")

if __name__ == "__main__":
    example_usage()
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result