import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --- configuration --- 
path_model = "/Random_forest.ipnyb"
path_output = "../reports"
os.makedirs(path_output, exist_ok = True)

# --- numeric and categorical features ---
numeric_features = ['net_income', 'net_cash_flow', 'roe', 'roa', 'ebitda', 'cumulation']
categorical_features = ['sector']

# --- loadGridSearchCV ---
grid_search = joblib.load(path_model)

# --- best pipeline ----
pipeline = grid_search.best_estimator_

model_rf = pipeline.named_steps["rf"]
preprocessor = pipeline.named_steps["preprocessing"]

cat_transformator = preprocessor.named_steps["cat"]
cat_encoder = cat_transformator.named_steps["encoder"]
categorical_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))

feature_names = numeric_feature_names + categorical_feature_names

importances = model_rf.feature_importances_

feature_importances_df = pd.DataFrame({
    "Features": feature_names,
    "Importance": importances 
}).sort_values(by="importances", ascending = False)

plt.figure(figsize=(10,8))
plt_data = feature_importances_df 
plt.bar(feature_importances_df ["Importances"],feature_importances_df["Features"])
plt.xlabel("Importances")
plt.title("Feature Importances")
plt.tight_layout()
path_plot = os.path.join(path_output, "feature_importance.png")
plt.savefig(path_plot)
plt.show()