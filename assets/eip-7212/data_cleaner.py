
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns

    def handle_missing_values(self, strategy='mean', fill_value=None):
        if strategy == 'mean' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].mean()
            )
        elif strategy == 'median' and self.numeric_columns.any():
            self.df[self.numeric_columns] = self.df[self.numeric_columns].fillna(
                self.df[self.numeric_columns].median()
            )
        elif strategy == 'mode' and self.categorical_columns.any():
            for col in self.categorical_columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        elif fill_value is not None:
            self.df = self.df.fillna(fill_value)
        return self

    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        for col in columns:
            if col in self.numeric_columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                self.df = self.df[
                    (self.df[col] >= lower_bound) & 
                    (self.df[col] <= upper_bound)
                ]
        return self

    def standardize_numeric(self):
        for col in self.numeric_columns:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                self.df[col] = (self.df[col] - mean) / std
        return self

    def encode_categorical(self, method='onehot'):
        if method == 'onehot' and self.categorical_columns.any():
            self.df = pd.get_dummies(self.df, columns=self.categorical_columns)
        elif method == 'label' and self.categorical_columns.any():
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in self.categorical_columns:
                self.df[col] = le.fit_transform(self.df[col])
        return self

    def get_cleaned_data(self):
        return self.df

def clean_dataset(df, missing_strategy='mean', outlier_removal=True, standardization=True):
    cleaner = DataCleaner(df)
    
    cleaner.handle_missing_values(strategy=missing_strategy)
    
    if outlier_removal:
        cleaner.remove_outliers_iqr()
    
    if standardization:
        cleaner.standardize_numeric()
    
    return cleaner.get_cleaned_data()