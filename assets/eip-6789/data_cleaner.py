
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif outlier_method == 'zscore':
            mean_val = cleaned_df[col].mean()
            std_val = cleaned_df[col].std()
            
            # Calculate z-scores and cap values beyond 3 standard deviations
            z_scores = np.abs((cleaned_df[col] - mean_val) / std_val)
            mask = z_scores > 3
            
            if mask.any():
                # Replace outliers with nearest non-outlier boundary
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                cleaned_df.loc[mask, col] = cleaned_df.loc[mask, col].clip(
                    lower=lower_bound, upper=upper_bound
                )
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"

# Example usage function
def demonstrate_cleaning():
    """Demonstrate the data cleaning functionality."""
    # Create sample data with missing values and outliers
    np.random.seed(42)
    data = {
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    df.loc[10:15, 'A'] = np.nan
    df.loc[20:25, 'B'] = np.nan
    
    # Introduce outliers
    df.loc[0, 'A'] = 100
    df.loc[1, 'B'] = -100
    
    print("Original dataframe shape:", df.shape)
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    
    print("\nCleaned dataframe shape:", cleaned.shape)
    print("Missing values after cleaning:")
    print(cleaned.isnull().sum())
    
    # Validate the cleaned dataframe
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
    
    return cleaned

if __name__ == "__main__":
    result = demonstrate_cleaning()
    print("\nCleaning demonstration completed successfully.")