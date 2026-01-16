import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and standardizing numeric columns.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(input_path)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = df[col].mean()
                elif missing_strategy == 'median':
                    fill_value = df[col].median()
                elif missing_strategy == 'drop':
                    df = df.dropna(subset=[col])
                    continue
                else:
                    raise ValueError(f"Unknown strategy: {missing_strategy}")
                
                df[col] = df[col].fillna(fill_value)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Original shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.Series: Boolean mask of outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'z']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        outliers = detect_outliers_iqr(cleaned_df, 'A')
        print(f"Outliers in column A: {outliers.sum()}")