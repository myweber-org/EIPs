import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val - min_val == 0:
        return dataframe[column].apply(lambda x: 0.0)
    normalized = (dataframe[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(dataframe, column):
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    if std_val == 0:
        return dataframe[column].apply(lambda x: 0.0)
    standardized = (dataframe[column] - mean_val) / std_val
    return standardized

def handle_missing_mean(dataframe, column):
    mean_val = dataframe[column].mean()
    filled_series = dataframe[column].fillna(mean_val)
    return filled_series

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2.0, 50),
        'feature_c': np.random.uniform(0, 1, 50)
    }
    df = pd.DataFrame(data)
    df.loc[5, 'feature_a'] = np.nan
    df.loc[10, 'feature_b'] = np.nan
    df.loc[15, 'feature_c'] = 500
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original sample data:")
    print(sample_df.head())
    
    cleaned_df = remove_outliers_iqr(sample_df, 'feature_c')
    print("\nData after removing outliers from feature_c:")
    print(cleaned_df.head())
    
    sample_df['feature_a_norm'] = normalize_minmax(sample_df, 'feature_a')
    sample_df['feature_b_std'] = standardize_zscore(sample_df, 'feature_b')
    sample_df['feature_a_filled'] = handle_missing_mean(sample_df, 'feature_a')
    
    print("\nProcessed data with new columns:")
    print(sample_df[['feature_a_norm', 'feature_b_std', 'feature_a_filled']].head())