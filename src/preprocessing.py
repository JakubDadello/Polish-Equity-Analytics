import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Load dataset ---
# all_data contains both input features (X) and target labels (Y)
all_data = pd.read_csv("../data/initial_labeling_data.csv")

# --- Split features and target ---
X = all_data.iloc[:, 2:-1]  # input features
Y = all_data.iloc[:, -1:]   # target labels

# --- Define numerical and categorical columns ---
numeric_features = ['net_income', 'net_cash_flow', 'roe', 'roa', 'ebitda', 'cumulation']
categorical_features = ['sector']

# --- Numerical preprocessing ---
# 1. Impute missing values with median
# 2. Standardize features to zero mean and unit variance
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# --- Categorical preprocessing ---
# 1. Impute missing values with the most frequent category
# 2. Apply One-Hot Encoding (ignore unknown categories in test set)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# --- Combine numerical and categorical preprocessing ---
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
