
import pandas as pd
import numpy as np

def clean_dataset(df, numeric_columns=None, fill_method='median', outlier_method='iqr', iqr_factor=1.5):
    """
    Clean a dataset by handling missing values and outliers for numeric columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    numeric_columns (list): List of numeric column names to process. If None, uses all numeric columns.
    fill_method (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    outlier_method (str): Method to handle outliers ('iqr' or 'zscore').
    iqr_factor (float): Factor for IQR method to determine outlier bounds.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    if numeric_columns is None:
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_columns:
        if col not in df_clean.columns:
            continue

        if fill_method == 'mean':
            fill_value = df_clean[col].mean()
        elif fill_method == 'median':
            fill_value = df_clean[col].median()
        elif fill_method == 'mode':
            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
        elif fill_method == 'drop':
            df_clean = df_clean.dropna(subset=[col])
            continue
        else:
            raise ValueError("fill_method must be 'mean', 'median', 'mode', or 'drop'")

        df_clean[col] = df_clean[col].fillna(fill_value)

        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            df_clean[col] = np.where((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
                                      np.nan, df_clean[col])
            df_clean[col] = df_clean[col].fillna(fill_value)

        elif outlier_method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            z_scores = (df_clean[col] - mean) / std
            df_clean[col] = np.where(np.abs(z_scores) > iqr_factor, np.nan, df_clean[col])
            df_clean[col] = df_clean[col].fillna(fill_value)

        else:
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")

    return df_clean

def example_usage():
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataset(df, numeric_columns=['A', 'B'], fill_method='median', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    example_usage()