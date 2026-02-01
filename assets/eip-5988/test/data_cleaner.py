
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    numeric_cols = ['A', 'B', 'C']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(f"Removed {len(sample_data) - len(result)} outliers")import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path, strategy='mean'):
    """
    Load a CSV file, clean missing values, and save the cleaned data.
    
    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to save the cleaned CSV file.
    strategy (str): Strategy for handling missing values ('mean', 'median', 'drop').
    
    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Original data shape: {df.shape}")
        
        if strategy == 'drop':
            df_cleaned = df.dropna()
        elif strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
            df_cleaned = df
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned
    
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    cleaned_df = clean_csv_data('raw_data.csv', 'cleaned_data.csv', strategy='mean')
    if cleaned_df is not None:
        print("Data cleaning completed successfully.")