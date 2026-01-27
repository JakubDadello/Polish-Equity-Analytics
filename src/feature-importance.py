import os
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load 
from preprocessing import numeric_features, categorical_features

# --- Load the serialized pipeline ---
pipeline = load("../models/pipeline_rf.joblib")

# --- Output directory configuration ---
path_output = "../reports"
os.makedirs(path_output, exist_ok=True)

# --- Feature names extraction ---
raw_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
feature_names = [n.split("__")[1] for n in raw_names]

# --- Feature importance extraction --- 
importances = pipeline.named_steps["rf"].feature_importances_

# Organize features and their scores into a DataFrame for easier sorting
feature_importances_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


# --- Visualization ---
plt.figure(figsize=(10, 8))

# Plotting the top 5 most impactful features for better readability
top_features = feature_importances_df.head(5)

plt.barh(top_features["Feature"], top_features["Importance"], color="skyblue")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Random Forest Feature Importances")

plt.tight_layout()

# --- Saving the results ---
path_plot = os.path.join(path_output, "feature_importance.png")
plt.savefig(path_plot)

plt.show()
