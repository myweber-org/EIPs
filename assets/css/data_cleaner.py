import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'mode', 'drop'.
    outlier_method (str): Method for detecting outliers.
                          Options: 'iqr', 'zscore'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
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
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif outlier_method == 'zscore':
            mean_val = cleaned_df[col].mean()
            std_val = cleaned_df[col].std()
            z_scores = (cleaned_df[col] - mean_val) / std_val
            cleaned_df[col] = cleaned_df[col].mask(np.abs(z_scores) > 3, mean_val)
    
    return cleaned_df

def validate_data(df, required_columns=None, unique_constraints=None):
    """
    Validate DataFrame structure and constraints.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    unique_constraints (list): List of columns that should have unique values.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if unique_constraints:
        for col in unique_constraints:
            if col in df.columns:
                duplicates = df[col].duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(
                        f"Column '{col}' has {duplicates} duplicate values"
                    )
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['warnings'].append(f"Column '{col}' contains infinite values")
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10, 20, None, 40, 1000],
        'category': ['A', 'B', 'A', None, 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_data(cleaned, required_columns=['id', 'value'])
    print("\nValidation Results:")
    print(validation)