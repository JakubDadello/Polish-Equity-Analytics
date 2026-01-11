import os 

# --- Import libraries and configure warnings ---
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Import necessary packages for preprocessing, modeling, and evaluation
from preprocessing import preprocessor, X, Y
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib 

# --- Define path to save trained model ---
# The pipeline will be saved here for future use
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "pipeline_rf.joblib")

# --- split data ---
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# --- flatten y to avoid DataConversionWarning ---
y_train = y_train.values.ravel()

# --- best parameters configuration ---
best_params = {
    "n_estimators": 200,
    "criterion": "entropy",
    "max_depth": None, 
    "random_state": 42
}

# --- define pipeline with Random Forest ---
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("rf", RandomForestClassifier(**best_params))
])

# --- train model ---
pipeline.fit(X_train, y_train)

# --- Save the trained pipeline ---
# --- Model along with preprocessing steps is saved using joblib for later use
joblib.dump(pipeline, model_path)