
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_method='iqr', fill_value=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode', 'constant').
    outlier_method (str): Method for outlier detection ('iqr' or 'zscore').
    fill_value: Value to use for constant imputation if strategy='constant'.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    # Handle missing values
    if strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    elif strategy == 'constant':
        if fill_value is not None:
            df_clean = df_clean.fillna(fill_value)
        else:
            raise ValueError("fill_value must be provided for constant imputation")
    else:
        raise ValueError("Invalid imputation strategy")

    # Handle outliers for numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
                                      df_clean[col].median(), df_clean[col])
        elif outlier_method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean[col] = np.where(z_scores > 3, df_clean[col].median(), df_clean[col])
        else:
            raise ValueError("Invalid outlier detection method")

    return df_clean

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = pd.DataFrame({
#         'A': [1, 2, np.nan, 4, 100],
#         'B': [5, np.nan, 7, 8, 9],
#         'C': ['x', 'y', 'z', 'x', 'y']
#     })
#     cleaned = clean_dataset(sample_data, strategy='median', outlier_method='iqr')
#     print("Original DataFrame:")
#     print(sample_data)
#     print("\nCleaned DataFrame:")
#     print(cleaned)