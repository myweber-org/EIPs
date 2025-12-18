import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df_clean.columns]
    
    # Handle missing values
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown missing strategy: {missing_strategy}")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    # Handle outliers
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            
            # Calculate z-scores
            z_scores = np.abs((df_clean[col] - mean_val) / std_val)
            
            # Cap values beyond 3 standard deviations
            df_clean.loc[z_scores > 3, col] = mean_val + 3 * std_val * np.sign(
                df_clean.loc[z_scores > 3, col] - mean_val
            )
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): Columns that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'A': [1, 2, np.nan, 4, 100],
#         'B': [5, 6, 7, 8, 9],
#         'C': [10, 11, 12, 13, 1000]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B', 'C'])
#     print(f"\nValidation: {is_valid} - {message}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not clean column '{col}': {e}")
    
    return cleaned_df

def save_cleaned_data(df, input_path, suffix='_cleaned'):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    input_path (str): Original file path
    suffix (str): Suffix to add to filename
    
    Returns:
    str: Path to saved file
    """
    if input_path.endswith('.csv'):
        output_path = input_path.replace('.csv', f'{suffix}.csv')
    else:
        output_path = f"{input_path}{suffix}.csv"
    
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.array([500, 600, 700, -300, -400])
        ])
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original data shape: {df.shape}")
    print(f"Original data statistics:\n{df['value'].describe()}")
    
    cleaned_df = clean_numeric_data(df, columns=['value'])
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned data statistics:\n{cleaned_df['value'].describe()}")
    
    outliers_removed = df.shape[0] - cleaned_df.shape[0]
    print(f"\nOutliers removed: {outliers_removed}")