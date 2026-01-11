import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
        
        if columns is None:
            columns = df.columns
        
        for col in columns:
            if col in df.columns:
                if df[col].isnull().any():
                    if strategy == 'mean' and np.issubdtype(df[col].dtype, np.number):
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == 'median' and np.issubdtype(df[col].dtype, np.number):
                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy == 'mode':
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
                    elif strategy == 'drop':
                        df.dropna(subset=[col], inplace=True)
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Path for output CSV file
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_missing_data(input_file, strategy='median')
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values.
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning to. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_clean[col].mean()
        elif strategy == 'median':
            fill_value = df_clean[col].median()
        elif strategy == 'mode':
            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
        elif strategy == 'fill_zero':
            fill_value = 0
        elif strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    df_norm = df.copy()
    
    if method == 'minmax':
        min_val = df_norm[column].min()
        max_val = df_norm[column].max()
        if max_val != min_val:
            df_norm[column] = (df_norm[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df_norm[column].mean()
        std_val = df_norm[column].std()
        if std_val != 0:
            df_norm[column] = (df_norm[column] - mean_val) / std_val
    
    return df_norm

if __name__ == "__main__":
    sample_data = {'A': [1, 2, np.nan, 4, 5],
                   'B': [np.nan, 2, 3, 4, 5],
                   'C': [1, 2, 3, 4, 5]}
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_missing_data(df, strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)