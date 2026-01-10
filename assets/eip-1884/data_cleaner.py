
import pandas as pd
import numpy as np

def clean_data(df):
    """
    Clean the input DataFrame by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values: fill numeric columns with median, categorical with mode
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in [np.float64, np.int64]:
            df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
        else:
            df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown', inplace=True)
    
    return df_cleaned

def validate_data(df):
    """
    Validate the cleaned DataFrame for any remaining issues.
    """
    validation_report = {}
    validation_report['total_rows'] = len(df)
    validation_report['total_columns'] = len(df.columns)
    validation_report['missing_values'] = df.isnull().sum().sum()
    validation_report['duplicate_rows'] = df.duplicated().sum()
    
    return validation_report

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'A': [1, 2, 2, 4, np.nan],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', np.nan]
    })
    
    cleaned_df = clean_data(sample_data)
    report = validate_data(cleaned_df)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\nValidation Report:")
    for key, value in report.items():
        print(f"{key}: {value}")