# --- hide warnings ---
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from preprocessing import preprocessor, X, Y
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# --- split data ---
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# --- flatten y to avoid DataConversionWarning ---
y_train = y_train.values.ravel()
y_test  = y_test.values.ravel()

# --- best_params from GridSearchCV ---
best_params = {
    "n_estimators": 200,
    "criterion": "entropy",
    "max_depth": 10, 
    "random_state": 42
}

# --- define pipeline with Random Forest ---
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("rf", RandomForestClassifier(**best_params))
])

# --- train model ---
pipeline.fit(X_train, y_train)

# --- predictions ---
y_pred = pipeline.predict(X_test)

# --- evaluation metrics ---
accuracy_values = accuracy_score(y_test, y_pred)
precision_values = precision_score(y_test, y_pred, average='macro')
recall_values = recall_score(y_test, y_pred, average='macro')
confusionmatrix_values = confusion_matrix(y_test, y_pred, labels=['low', 'middle', 'high'])
