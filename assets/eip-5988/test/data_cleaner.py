import pandas as pd
import numpy as np
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def detect_outliers_iqr(self, column, threshold=1.5):
        if column not in self.numeric_columns:
            raise ValueError(f"Column '{column}' is not numeric")
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        return outliers.index.tolist()
    
    def remove_outliers(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        outlier_indices = set()
        for col in columns:
            if col in self.numeric_columns:
                indices = self.detect_outliers_iqr(col, threshold)
                outlier_indices.update(indices)
        
        cleaned_df = self.df.drop(index=list(outlier_indices))
        return cleaned_df
    
    def impute_missing_median(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        imputed_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns and imputed_df[col].isnull().any():
                median_val = imputed_df[col].median()
                imputed_df[col].fillna(median_val, inplace=True)
        
        return imputed_df
    
    def standardize_data(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        standardized_df = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean = standardized_df[col].mean()
                std = standardized_df[col].std()
                if std > 0:
                    standardized_df[col] = (standardized_df[col] - mean) / std
        
        return standardized_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': self.numeric_columns,
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature_a'] = np.nan
    df.loc[[5, 25, 50], 'feature_a'] = 500
    
    cleaner = DataCleaner(df)
    print("Data Summary:", cleaner.get_summary())
    
    outliers = cleaner.detect_outliers_iqr('feature_a')
    print(f"Outliers in feature_a: {outliers}")
    
    cleaned = cleaner.remove_outliers(['feature_a'])
    print(f"Cleaned shape: {cleaned.shape}")
    
    imputed = cleaner.impute_missing_median()
    print(f"Missing values after imputation: {imputed.isnull().sum().sum()}")
    
    standardized = cleaner.standardize_data()
    print(f"Standardized mean: {standardized['feature_a'].mean():.2f}")
    
    return cleaned

if __name__ == "__main__":
    result_df = example_usage()
    print(f"Final cleaned dataframe shape: {result_df.shape}")