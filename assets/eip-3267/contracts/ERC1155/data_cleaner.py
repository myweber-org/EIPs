
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

def main():
    data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10]}
    df = pd.DataFrame(data)
    print("Original data:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['values'])
    print("\nCleaned data:")
    print(cleaned_df)

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def normalize_column(df, column):
    """Normalize column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    """Apply outlier removal and normalization to numeric columns."""
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers(df, col)
            df = normalize_column(df, col)
    return df

if __name__ == "__main__":
    data = load_data('sample_data.csv')
    numeric_cols = ['age', 'income', 'score']
    cleaned_data = clean_data(data, numeric_cols)
    cleaned_data.to_csv('cleaned_data.csv', index=False)