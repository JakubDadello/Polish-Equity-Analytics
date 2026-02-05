import pandas as pd 
from fastapi import FastAPI
from joblib import load 
from schemas import Model_Request

# --- Initialize FastAPI app with metadata for automatic Swagger/ReDoc documentation ---
app = FastAPI()

# --- Load the serialized model pipeline into memory during startup ---
MODEL_PATH = "../models/pipeline_rf.joblib"
model = load(MODEL_PATH )

@app.post("/run_pipeline", )
async def classify (item: Model_Request):
   """
    Predicts the investment rating based on the provided financial features.
    
    Args:
        item (Model_Request): Validated Pydantic model containing input features.
        
    Returns:
        dict: Predicted investment rating category.
    """
   
   # --- Convert incoming Pydantic schema to DataFrame for model compatibility ---
   input = pd.DataFrame([item.model_dump()])

   # --- Perform inference and extract the first element from the prediction array ---
   output = model.predict(input)[0]

   # --- Return as JSON-serializable dictionary ---
   return {
      "investment_rating": output
   }
