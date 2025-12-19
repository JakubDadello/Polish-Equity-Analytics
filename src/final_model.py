import sys
import os
sys.path.append(os.path.abspath(".."))

# --- hide warnings ---
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

import numpy as np
import pandas as pd
from src.preprocessing import preprocessor, X, Y
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# --- split data ---
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# --- flatten y to avoid DataConversionWarning ---
y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

# --- define pipeline with Random Forest ---
chain = Pipeline([
    ("preprocessing", preprocessor),
    ("rf", RandomForestClassifier(random_state=42))
])

# --- parameters for GridSearchCV ---
param_grid = {
    "rf__n_estimators": [50, 100, 200],
    "rf__criterion": ["gini", "entropy"],
    "rf__max_depth": [None, 5, 10]
}

# --- GridSearchCV object ---
grid_search = GridSearchCV(chain, param_grid, scoring='accuracy', cv=5)

# --- train model ---
grid_search.fit(X_train, y_train)

# --- predictions ---
y_pred = grid_search.predict(X_test)

# --- evaluation metrics ---
accuracy_values = accuracy_score(y_test, y_pred)
precision_values = precision_score(y_test, y_pred, average='macro')
recall_values = recall_score(y_test, y_pred, average='macro')
confusionmatrix_values = confusion_matrix(y_test, y_pred, labels=['low', 'middle', 'high'])