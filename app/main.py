import pandas as pd 
from fastapi import FastAPI
from joblib import load 
from schemas import Model_Request

app = FastAPI()

model = load("../models/pipeline_rf.joblib")

@app.post("/run_pipeline", )
async def classify (item: Model_Request):
   input = pd.DataFrame([item.model_dump()])
   output = model.predict(input)[0]

   return {
      "investment_rating": output
   }
