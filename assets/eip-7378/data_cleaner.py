import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): Specific columns to check for missing values, defaults to all columns
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns
    
    for column in columns_to_check:
        if column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            
            if missing_count > 0:
                print(f"Column '{column}' has {missing_count} missing values")
                
                if fill_missing == 'drop':
                    df_clean = df_clean.dropna(subset=[column])
                elif fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
                    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
                    df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                elif fill_missing == 'mode':
                    mode_value = df_clean[column].mode()
                    if not mode_value.empty:
                        df_clean[column] = df_clean[column].fillna(mode_value.iloc[0])
                else:
                    # For non-numeric columns or unknown method, fill with most frequent value
                    most_frequent = df_clean[column].value_counts().index[0]
                    df_clean[column] = df_clean[column].fillna(most_frequent)
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate data quality after cleaning.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): Columns that must be present
    numeric_columns (list): Columns that should be numeric
    
    Returns:
    dict: Validation results
    """
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'validation_passed': True
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['missing_columns'] = missing_columns
            validation_results['validation_passed'] = False
    
    # Check for missing values
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        validation_results['missing_values'][column] = missing_count
        validation_results['data_types'][column] = str(df[column].dtype)
        
        if missing_count > 0:
            validation_results['validation_passed'] = False
    
    # Validate numeric columns
    if numeric_columns:
        for column in numeric_columns:
            if column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    validation_results['validation_passed'] = False
                    if 'non_numeric_columns' not in validation_results:
                        validation_results['non_numeric_columns'] = []
                    validation_results['non_numeric_columns'].append(column)
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'id': [1, 2, 3, 4, 5, 5, 6],
#         'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
#         'age': [25, 30, None, 35, 40, 40, 45],
#         'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\n" + "="*50 + "\n")
#     
#     # Clean the data
#     cleaned_df = clean_dataset(df, fill_missing='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     # Validate the cleaned data
#     validation = validate_data(cleaned_df, 
#                               required_columns=['id', 'name', 'age', 'score'],
#                               numeric_columns=['age', 'score'])
#     print("\nValidation Results:")
#     for key, value in validation.items():
#         print(f"{key}: {value}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def generate_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1, 200)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    sample_df = generate_sample_data()
    numeric_cols = ['feature_a', 'feature_b', 'feature_c']
    
    cleaned_data = clean_dataset(
        sample_df, 
        numeric_columns=numeric_cols,
        outlier_method='iqr',
        normalize_method='zscore'
    )
    
    print(f"Original shape: {sample_df.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Removed {len(sample_df) - len(cleaned_data)} outliers")
    print("\nCleaned data summary:")
    print(cleaned_data.describe())