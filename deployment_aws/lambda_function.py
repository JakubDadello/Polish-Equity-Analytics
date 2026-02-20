import os 
import boto3
import joblib 

import boto3
import joblib
import os


S3_BUCKET = "pqa-final-model"
MODEL_KEY = "pipeline_rf.joblib"

s3 = boto3.client("s3")
model = None

def load_model():
    
    local_path = "/tmp/pipeline_rf.joblib"

    if not os.path.exists(local_path):
        s3.download_file(S3_BUCKET, MODEL_KEY, local_path)

    return joblib.load(local_path)


def lambda_handler(event, context):
    global model

    if model is None:
        model = load_model()

    data = event.get("data", [[1, 2, 3]])

    prediction = model.predict(data)

    return {
        "input": data,
        "prediction": prediction.tolist()
    }