# --- Standard library imports ---
import os       # OS interactions and environment variables
import logging  # Application logging and diagnostics


# --- Import libraries and configure warnings ---
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# --- Import necessary packages for preprocessing, modeling, and evaluation ---
from db_utils import load_data
from preprocessing import preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Machine Learning & Experiment Tracking ---
import mlflow
import mlflow.sklearn
import joblib  # Model persistence


# --- Logging configuration ---
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_model(MODEL_PATH: str):

    if os.path.exists(MODEL_PATH):
        logging.info(f"Model already exists")
        return 
         
    # --- The pipeline will be saved here for future use --- 
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    # --- Load data from Postgres ---
    df = load_data("SELECT * FROM initial_labeling_data ORDER BY id" )

    # --- Split features and target ---
    X = df.iloc[:, 2:-1]  # input features
    Y = df.iloc[:, -1:]   # target labels

    # --- split data ---
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # --- flatten y to avoid DataConversionWarning ---
    Y_train = Y_train.values.ravel()
    Y_test = Y_test.values.ravel()

    # --- best parameters configuration ---
    best_params = {
        "n_estimators": 200,
        "criterion": "entropy",
        "max_depth": None, 
        "random_state": 42
        }

    # --- define pipeline with Random Forest ---
    pipeline = Pipeline([
        ("preprocessing", preprocessor()),
        ("rf", RandomForestClassifier(**best_params))
    ])

    # --- train model ---
    pipeline.fit(X_train, Y_train)

    # --- predictions ---
    Y_pred = pipeline.predict(X_test)

    # --- evaluation ---
    accuracy = accuracy_score(Y_test, Y_pred)

    # --- Save the trained pipeline ---
    joblib.dump(pipeline, MODEL_PATH)

    # --- MLflow experiment ---
    mlflow.set_experiment("")
        
    with mlflow.start_run():

        # 1. Log hyperparameters found during tuning
        mlflow.log_params(best_params)

        # 2. Log the performance metric 
        mlflow.log_metric("accuracy", accuracy)

        # 3. Log the model artifact with a signature 
        mlflow.sklearn.log_model(pipeline, "model")

    return accuracy

# --- Execute the script ---
if __name__ == "__main__":
    train_model()



        
    
