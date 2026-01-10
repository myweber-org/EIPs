
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing columns
    with excessive missing data.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Threshold for dropping columns (0.0 to 1.0)
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    df = pd.read_csv(filepath)
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing percentage per column
    missing_percent = df.isnull().sum() / len(df)
    
    # Drop columns with missing values above threshold
    columns_to_drop = missing_percent[missing_percent > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    print(f"Dropped {len(columns_to_drop)} columns with >{drop_threshold*100}% missing values")
    
    # Fill remaining missing values based on method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_method == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    elif fill_method == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        most_frequent = df[col].mode()
        if not most_frequent.empty:
            df[col] = df[col].fillna(most_frequent.iloc[0])
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Total missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned dataframe
    output_path (str): Path for output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_method='median', drop_threshold=0.3)
        save_cleaned_data(cleaned_df, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")import pandas as pd

def remove_duplicate_rows(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: Input is not a DataFrame")
        return False
    
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        dataframe (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['object']).columns
    
    for column in columns:
        if column in dataframe.columns:
            try:
                dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
            except Exception as e:
                print(f"Could not convert column {column}: {e}")
    
    return dataframe
import pandas as pd
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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned.head())import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name):
    """
    Clean a numeric column by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True