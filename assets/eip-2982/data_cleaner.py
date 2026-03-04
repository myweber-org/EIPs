import numpy as np
import pandas as pd
from scipy import stats

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_mean(df, column):
    mean_val = df[column].mean()
    df[column].fillna(mean_val, inplace=True)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if cleaned_df[col].isnull().sum() > 0:
            cleaned_df = handle_missing_mean(cleaned_df, col)
        cleaned_df = remove_outliers_iqr(cleaned_df, col)
        cleaned_df = normalize_minmax(cleaned_df, col)
        cleaned_df = standardize_zscore(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature3': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    df = pd.DataFrame(sample_data)
    cleaned = clean_dataset(df, ['feature1', 'feature2', 'feature3'])
    print("Original dataset shape:", df.shape)
    print("Cleaned dataset shape:", cleaned.shape)
    print("\nCleaned dataset head:")
    print(cleaned.head())import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', output_filepath=None):
    """
    Load a CSV file, clean missing values, and optionally save the cleaned data.
    
    Args:
        filepath (str): Path to the input CSV file.
        fill_strategy (str): Strategy for filling missing values. 
                             Options: 'mean', 'median', 'mode', 'zero', 'drop'.
        output_filepath (str, optional): Path to save the cleaned CSV file.
                                         If None, the cleaned DataFrame is returned.
    
    Returns:
        pandas.DataFrame or None: Cleaned DataFrame if output_filepath is None, else None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}. Shape: {df.shape}")
        
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Found {missing_before} missing values.")
            
            if fill_strategy == 'drop':
                df_cleaned = df.dropna()
                print("Dropped rows with missing values.")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if fill_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df[col].mode()[0]
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
                        
                        df[col].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in column '{col}' with {fill_strategy} value: {fill_value:.2f}")
                
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    if df[col].isnull().any():
                        df[col].fillna('Unknown', inplace=True)
                        print(f"Filled missing values in non-numeric column '{col}' with 'Unknown'")
            
            missing_after = df.isnull().sum().sum()
            print(f"Missing values after cleaning: {missing_after}")
        else:
            print("No missing values found in the dataset.")
            df_cleaned = df
        
        if output_filepath:
            df_cleaned.to_csv(output_filepath, index=False)
            print(f"Cleaned data saved to {output_filepath}")
            return None
        else:
            return df_cleaned
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of column names that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    # Example usage
    input_csv = "sample_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Create a sample dataset for demonstration
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', np.nan, 'z', 'w'],
        'D': [10.5, 20.3, 15.7, np.nan, 25.1]
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(input_csv, index=False)
    print(f"Created sample data at {input_csv}")
    
    # Clean the data
    cleaned_df = clean_csv_data(
        filepath=input_csv,
        fill_strategy='mean',
        output_filepath=output_csv
    )
    
    # Validate the cleaned data
    if cleaned_df is not None:
        validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C', 'D'])import pandas as pd
import numpy as np

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
        df = normalize_minmax(df, col)
    
    df.to_csv('cleaned_data.csv', index=False)
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv')
    print(f"Cleaned dataset shape: {cleaned_df.shape}")