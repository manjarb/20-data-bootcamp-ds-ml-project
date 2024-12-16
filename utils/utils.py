import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# List of categorical columns for feature engineering
numerical_columns = ['total_female', 'total_male', 'total_people', 'total_nights', 'night_mainland', 'night_zanzibar']
total_columns_to_check = ['total_cost', 'total_people', 'total_nights']

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
    # if 'total_cost' in df.columns:
    #     # Log-transformed column for total_cost
    #     df['total_cost_log'] = np.log1p(df['total_cost'])  # Adds 1 to avoid log(0)
    #     # Capped version of total_cost at the 99th percentile
    #     cap_value = df['total_cost'].quantile(0.99)
    #     df['total_cost_capped'] = np.where(df['total_cost'] > cap_value, cap_value, df['total_cost'])

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

def evaluate_model(y_true, y_pred):
    """
    Evaluate a regression model's performance.
    
    Parameters:
    - y_true (array-like): Actual target values.
    - y_pred (array-like): Predicted target values.
    
    Returns:
    - dict: A dictionary containing evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R² Score": r2
    }

    # Print metrics for quick reference
    print("Validation Set Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

# Identify and Remove Outliers
def remove_outliers(df, columns, method='iqr'):
    """
    Removes outliers from specified columns based on the chosen method.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): The columns to check for outliers.
    - method (str): The method to identify outliers ('iqr' or 'std').
    
    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    elif method == 'std':
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def apply_scaling(df, numerical_columns):
    """
    Apply Standard Scaling to the specified numerical columns in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numerical_columns (list): List of numerical column names to scale.
    
    Returns:
    - pd.DataFrame: The transformed DataFrame with scaled numerical columns.
    - StandardScaler: The scaler object used for scaling (for reproducibility or inverse transform if needed).
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_columns])
    df[numerical_columns] = scaled_data
    return df, scaler