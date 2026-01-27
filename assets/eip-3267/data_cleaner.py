
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

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('sample_data.csv', ['age', 'salary', 'score'])
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print(f"Data cleaned. Original shape: {pd.read_csv('sample_data.csv').shape}, Cleaned shape: {cleaned_df.shape}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (DataFrame): The dataset.
    column (str): The column name to clean.
    
    Returns:
    DataFrame: The dataset with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (DataFrame): The dataset.
    column (str): The column name.
    
    Returns:
    dict: A dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 14, 15, 16, 17, 18, 100]
    })
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("Original data:")
    print(sample_data)
    print("\nCleaned data:")
    print(cleaned_data)
    
    stats = calculate_basic_stats(cleaned_data, 'values')
    print("\nBasic statistics for cleaned data:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")