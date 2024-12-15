import pandas as pd
import numpy as np

# List of categorical columns for feature engineering
categorical_columns = [
    'country', 
    'age_group', 
    'travel_with', 
    'purpose',
    'main_activity', 
    'info_source', 
    'tour_arrangement', 
    'payment_mode',
    'first_trip_tz'
]

full_categorical_columns = [
    'country',
    'age_group',
    'travel_with',
    'purpose',
    'main_activity',
    'info_source',
    'tour_arrangement',
    'package_transport_int',
    'package_accomodation',
    'package_food',
    'package_transport_tz',
    'package_sightseeing',
    'package_guided_tour',
    'package_insurance',
    'payment_mode',
    'first_trip_tz',
    'most_impressing'
]

# Function to apply one-hot encoding to the DataFrame
def apply_one_hot_encoding(df, target_column, categorical_columns):
    """
    Apply one-hot encoding to the specified categorical columns.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the target column to exclude from encoding.
    - categorical_columns (list): List of categorical column names to encode.
    
    Returns:
    - pd.DataFrame: The transformed DataFrame with one-hot encoding applied.
    """
    
    # Drop the target column only if it exists in the DataFrame
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)


def clean_and_preprocess_data(df):
    """
    Cleans and preprocesses the tourism survey data for ML model usage.
    Args:
        df (pd.DataFrame): Input raw DataFrame.
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    # 1. Fill numerical columns with the median
    if 'total_female' in df.columns:
        df['total_female'] = df['total_female'].fillna(df['total_female'].median())
    if 'total_male' in df.columns:
        df['total_male'] = df['total_male'].fillna(df['total_male'].median())

    # 2. Fill categorical columns with 'Missing'
    if 'travel_with' in df.columns:
        df['travel_with'] = df['travel_with'].fillna('Missing')
    if 'most_impressing' in df.columns:
        df['most_impressing'] = df['most_impressing'].fillna('Missing')

    # 3. Convert column names to lowercase, replace spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 4. Handle outliers
    if 'total_cost' in df.columns:
        # Log-transformed column for total_cost
        df['total_cost_log'] = np.log1p(df['total_cost'])  # Adds 1 to avoid log(0)
        # Capped version of total_cost at the 99th percentile
        cap_value = df['total_cost'].quantile(0.99)
        df['total_cost_capped'] = np.where(df['total_cost'] > cap_value, cap_value, df['total_cost'])

    # 5. Target/Ordinal encoding for 'country'
    if 'country' in df.columns:
        df['country_encoded'] = df['country'].astype('category').cat.codes

    # 6. Create new features (feature engineering)
    if 'total_female' in df.columns and 'total_male' in df.columns:
        df['total_people'] = df['total_female'] + df['total_male']  # Total group size
    if 'night_mainland' in df.columns and 'night_zanzibar' in df.columns:
        df['total_nights'] = df['night_mainland'] + df['night_zanzibar']  # Total nights spent

    # 7. Drop unnecessary columns
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    return df


def align_features(X_train, X_test):
    """
    Aligns the test dataset's features to match the training dataset's features.

    Parameters:
    - X_train (pd.DataFrame): The training dataset used to fit the model.
    - X_test (pd.DataFrame): The test dataset to be aligned.

    Returns:
    - pd.DataFrame: The aligned test dataset with the same features as the training dataset.
    """
    # Find missing columns in the test data
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0  # Add missing columns with 0 values

    # Remove extra columns from the test data
    extra_cols = set(X_test.columns) - set(X_train.columns)
    X_test = X_test.drop(columns=extra_cols, errors='ignore')

    # Reorder columns to match training data
    X_test = X_test[X_train.columns]

    return X_test