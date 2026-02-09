import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to apply cleaning to, if None applies to all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    else:
        for col in columns:
            if col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if strategy == 'mean':
                        fill_value = df_clean[col].mean()
                    elif strategy == 'median':
                        fill_value = df_clean[col].median()
                    elif strategy == 'mode':
                        fill_value = df_clean[col].mode()[0]
                    else:
                        raise ValueError(f"Unsupported strategy: {strategy}")
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to standardize
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    from sklearn.preprocessing import StandardScaler
    
    df_standardized = df.copy()
    
    if columns is None:
        numeric_cols = df_standardized.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    scaler = StandardScaler()
    df_standardized[columns] = scaler.fit_transform(df_standardized[columns])
    
    return df_standardized

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
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
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Args:
        filepath: Path to the CSV file
        missing_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        columns_to_drop: List of column names to drop (optional)
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isnull().any():
                if missing_strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif missing_strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif missing_strategy == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)
                elif missing_strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
        
        for column in df.select_dtypes(include=['object']).columns:
            if df[column].isnull().any():
                df[column].fillna('Unknown', inplace=True)
        
        df = df.reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def export_cleaned_data(df, output_path):
    """
    Export cleaned DataFrame to CSV.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path for output CSV file
    """
    if df is not None and not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data exported to {output_path}")
    else:
        print("No data to export")