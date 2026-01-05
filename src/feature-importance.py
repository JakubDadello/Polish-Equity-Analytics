import os
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import numeric_features, categorical_features
from final_model import pipeline

# --- output ---
path_output = "../reports"
os.makedirs(path_output, exist_ok=True)

# --- feature names ---
feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()

# --- feature importance --- 
importances = pipeline.named_steps["rf"].feature_importances_

feature_importances_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


# --- plot ---
plt.figure(figsize=(10, 8))
top_features = feature_importances_df.head(5)

plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Random Forest Feature Importances")
plt.style.use("dark_background")
plt.tight_layout()

path_plot = os.path.join(path_output, "feature_importance.png")
plt.savefig(path_plot)
plt.show()
