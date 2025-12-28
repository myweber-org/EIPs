import numpy as np
import pandas as pd

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask]
        return self
        
    def normalize_column(self, column, method='minmax'):
        if column not in self.numeric_columns:
            raise ValueError(f"Column {column} is not numeric")
            
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[column] = (self.df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[column] = (self.df[column] - mean_val) / std_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        return self
        
    def fill_missing(self, column, strategy='mean'):
        if column not in self.df.columns:
            raise ValueError(f"Column {column} does not exist")
            
        if strategy == 'mean' and column in self.numeric_columns:
            fill_value = self.df[column].mean()
        elif strategy == 'median' and column in self.numeric_columns:
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        else:
            fill_value = 0
            
        self.df[column] = self.df[column].fillna(fill_value)
        return self
        
    def get_cleaned_data(self):
        return self.df.copy()
        
    def summary(self):
        summary_data = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().sum()
        }
        return summary_data

def example_usage():
    np.random.seed(42)
    data = {
        'A': np.random.normal(100, 15, 100),
        'B': np.random.uniform(0, 1, 100),
        'C': np.random.randint(1, 100, 100)
    }
    df = pd.DataFrame(data)
    df.loc[10:15, 'A'] = np.nan
    df.loc[5, 'B'] = 999
    
    cleaner = DataCleaner(df)
    cleaner.remove_outliers_iqr('A')
    cleaner.normalize_column('B', method='minmax')
    cleaner.fill_missing('A', strategy='mean')
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Summary: {cleaner.summary()}")
    
    return cleaned_df

if __name__ == "__main__":
    example_usage()
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled missing values in column '{col}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col].fillna('Unknown', inplace=True)
            print(f"Filled missing values in categorical column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty.")
        return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, None],
        'category': ['A', 'B', 'B', None, 'A', 'C'],
        'score': [85, 92, 92, 78, None, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation_result = validate_data(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"\nData validation result: {validation_result}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using the Interquartile Range method.
    Returns filtered DataFrame.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    Returns filtered DataFrame.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data to [0,1] range using Min-Max scaling.
    Returns normalized Series.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    Returns normalized Series.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    normalized = (data[column] - mean_val) / std_val
    return normalized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values using specified strategy.
    Returns DataFrame with imputed values.
    """
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif strategy == 'drop':
        return data.dropna()
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    Returns cleaned DataFrame.
    """
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data = handle_missing_values(cleaned_data[[col]], strategy=missing_strategy)
            
            if outlier_method == 'iqr':
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
            elif outlier_method == 'zscore':
                cleaned_data = remove_outliers_zscore(cleaned_data, col)
            
            if normalize_method == 'minmax':
                cleaned_data[col] = normalize_minmax(cleaned_data, col)
            elif normalize_method == 'zscore':
                cleaned_data[col] = normalize_zscore(cleaned_data, col)
    
    return cleaned_data