import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if missing_strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif missing_strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif missing_strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                cleaned_df.dropna(subset=[column], inplace=True)
    
    # Handle outliers using z-score method
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
        outliers = z_scores > outlier_threshold
        if outliers.any():
            # Cap outliers to threshold * standard deviation
            upper_bound = cleaned_df[column].mean() + outlier_threshold * cleaned_df[column].std()
            lower_bound = cleaned_df[column].mean() - outlier_threshold * cleaned_df[column].std()
            cleaned_df[column] = np.clip(cleaned_df[column], lower_bound, upper_bound)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    np.random.seed(42)
    sample_data = {
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    }
    
    # Introduce missing values
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[10:15, 'A'] = np.nan
    sample_df.loc[20:25, 'B'] = np.nan
    
    # Introduce outliers
    sample_df.loc[0, 'C'] = 100
    sample_df.loc[1, 'C'] = -100
    
    print("Original DataFrame shape:", sample_df.shape)
    print("Missing values:\n", sample_df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataframe(sample_df, missing_strategy='mean', outlier_threshold=3)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Missing values after cleaning:\n", cleaned_df.isnull().sum())
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'], min_rows=50)
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    print(f"Cleaning complete. Final dataset shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame for required columns and minimum rows.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
    
    if df.shape[0] < min_rows:
        print(f"Validation failed: Insufficient rows. Have {df.shape[0]}, need at least {min_rows}")
        return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, None, 30, 40, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'x', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nCleaning dataset...")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='median')
    
    print("\nCleaned dataset:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B', 'C'], min_rows=3)
    print(f"\nData is valid: {is_valid}")import re
import string

def clean_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Removing punctuation.
    3. Removing extra whitespace.
    4. Removing numbers.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def normalize_whitespace(text):
    """
    Normalizes whitespace in the input text by replacing
    any sequence of whitespace characters with a single space.
    """
    if not isinstance(text, str):
        return ""
    return ' '.join(text.split())

def remove_special_characters(text, keep_chars=''):
    """
    Removes special characters from the input text.
    The `keep_chars` parameter allows specifying characters to preserve.
    """
    if not isinstance(text, str):
        return ""

    # Define characters to remove (all except alphanumeric and kept chars)
    pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
    return re.sub(pattern, '', text)