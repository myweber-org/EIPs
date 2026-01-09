
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    """
    Remove outliers using IQR method
    """
    df_clean = dataframe.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(dataframe, columns, method='minmax'):
    """
    Normalize data using specified method
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if method == 'minmax':
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def clean_dataset(file_path, output_path=None):
    """
    Main function to clean dataset
    """
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Original shape: {df.shape}")
        
        df_clean = remove_outliers_iqr(df, numeric_cols)
        print(f"After outlier removal: {df_clean.shape}")
        
        df_normalized = normalize_data(df_clean, numeric_cols, method='minmax')
        
        if output_path:
            df_normalized.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df_normalized
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_dataset('raw_data.csv', 'cleaned_data.csv')