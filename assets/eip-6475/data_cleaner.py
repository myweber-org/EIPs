import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing ratio above this threshold
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = missing_percentage[missing_percentage > drop_threshold * 100].index
    df = df.drop(columns=columns_to_drop)
    
    if len(columns_to_drop) > 0:
        print(f"Dropped columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == 'zero':
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
    elif fill_strategy == 'mode':
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after cleaning")
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Total missing values removed: {original_shape[0]*original_shape[1] - df.shape[0]*df.shape[1]}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame
    output_path (str): Path to save the cleaned CSV
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return True
    return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
    
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    if columns is None:
        columns = df.columns
    
    for column in columns:
        if column in df.columns:
            if df[column].isnull().any():
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    df[column].fillna(df[column].median(), inplace=True)
                elif strategy == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
                else:
                    print(f"Warning: Cannot apply {strategy} to column {column}")
    
    return df

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pandas.Series: Boolean series indicating outliers
    """
    if column not in df.columns:
        print(f"Error: Column {column} not found in dataframe")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Error: Column {column} is not numeric")
        return None
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned dataframe
        output_path (str): Path to save the cleaned data
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    else:
        print("Error: No data to save")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_missing_data(input_file, strategy='median')
    
    if cleaned_df is not None:
        for col in cleaned_df.select_dtypes(include=[np.number]).columns:
            outliers = detect_outliers_iqr(cleaned_df, col)
            if outliers is not None and outliers.any():
                print(f"Found {outliers.sum()} outliers in column {col}")
        
        save_cleaned_data(cleaned_df, output_file)