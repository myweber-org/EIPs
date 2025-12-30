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
    if max_val != min_val:
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        for col in df.select_dtypes(include=[np.number]).columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned shape: {df.shape}")
        print(f"Data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = df.columns.tolist()
    
    def remove_outliers_iqr(self, column, multiplier=1.5):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        self.df = self.df[mask].reset_index(drop=True)
        return self
    
    def normalize_column(self, column, method='minmax'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if method == 'minmax':
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            if max_val != min_val:
                self.df[f"{column}_normalized"] = (self.df[column] - min_val) / (max_val - min_val)
            else:
                self.df[f"{column}_normalized"] = 0.5
        
        elif method == 'zscore':
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            if std_val > 0:
                self.df[f"{column}_normalized"] = (self.df[column] - mean_val) / std_val
            else:
                self.df[f"{column}_normalized"] = 0
        
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
        
        return self
    
    def fill_missing(self, column, strategy='mean'):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if strategy == 'mean':
            fill_value = self.df[column].mean()
        elif strategy == 'median':
            fill_value = self.df[column].median()
        elif strategy == 'mode':
            fill_value = self.df[column].mode()[0]
        elif isinstance(strategy, (int, float)):
            fill_value = strategy
        else:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or a numeric value")
        
        self.df[column] = self.df[column].fillna(fill_value)
        return self
    
    def get_cleaned_data(self):
        return self.df
    
    def get_summary(self):
        summary = {
            'original_rows': len(self.df) if hasattr(self, '_original_rows') else 'Not tracked',
            'current_rows': len(self.df),
            'columns': self.df.columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 5), 'age'] = np.nan
    df.loc[np.random.choice(100, 5), 'income'] = np.nan
    
    df.loc[0, 'income'] = 1000000
    df.loc[1, 'age'] = 150
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original data shape:", sample_df.shape)
    print("\nMissing values:")
    print(sample_df.isnull().sum())
    
    cleaner = DataCleaner(sample_df)
    
    cleaner.fill_missing('age', 'median')
    cleaner.fill_missing('income', 'mean')
    
    cleaner.remove_outliers_iqr('income')
    cleaner.remove_outliers_iqr('age')
    
    cleaner.normalize_column('score', 'minmax')
    cleaner.normalize_column('income', 'zscore')
    
    cleaned_df = cleaner.get_cleaned_data()
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned data summary:")
    print(cleaned_df.describe())