import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_processed_data():
    """Fetches, cleans, splits, and preprocesses the House Prices dataset."""
    print("Fetching dataset...")
    house_prices = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    df = house_prices.frame
    
    # 1. Clean up mostly empty columns
    cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 2. Separate Features and Target (Applying log transformation)
    X = df.drop('SalePrice', axis=1)
    y = np.log1p(df['SalePrice']) 
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Identify columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    # 5. Build Pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # 6. Transform Data
    print("Applying transformations...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Data preprocessing complete!")
    return X_train_processed, X_test_processed, y_train.values, y_test.values, preprocessor