# Creating a FastAPI application for serving the ML model
# trained earlier on into a web service that can be used by
# anyone or any system by calling over the HTTP

"""Execution order:
1. Imports (FastAPI, pandas, boto3, the inference function)
2. Configuration step (Environment variables and bucket and region)
3. S3 utility (load_from_s3)
4. Downlaod and load the model and the required artifacts such as
(MODEL_PATH, TRAIN_FE_PATH)
5. Read schema (TRAIN_FEATURE_COLUMNS)
6. Create the FastAPI application (app = FastAPI)
7. Declare the endpoints (/, /health, /predict, /run_batch,
/latest_predictions)"""

from fastapi import FastAPI            # API web framework
from pathlib import Path               # File paths handling
from typing import List, Dict, Any     # For type hints
# (clarity in endpoints)
import pandas as pd                    # Incoming JSON files
# as Dataframes
import boto3, os                       # AWS SDK for Python
# and other environment variables

from src.inference_pipeline.inference import predict


# Configuration

S3_BUCKET = os.getenv("S3_BUCKET", "housing-prices-regression-ml-end-to-end")
REGION = os.getenv("AWS_REGION", "eu-west-2")
s3 = boto3.client("s3", region_name=REGION)


# Making sure that the application always has the latest model
# and the data locally by avoiding the re-downloads on each start-up

# Loader Function
def load_from_s3(key, local_path):
    """Download from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)


# Paths
# Downloads the model and the training features from S3 if not
# cached already
MODEL_PATH = Path(load_from_s3("models/xgb_best_model.pkl", "models/xgb_best_model.pkl"))
TRAIN_FE_PATH = Path(load_from_s3("processed/feature_engineered_train.csv", "data/processed/feature_engineered_train.csv"))


# Load the expected training features for the alignment
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns
                             if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None


# Application
#Instantiates the FastAPI
app = FastAPI(title="Housing Prices Regression API")


# Landing endpoint for checking if the application is live or not
@app.get("/")
def root():
    return {"message": "Housing Prices Regression API is running"}

# Health check along with checking the model exists, returns
# status information (such as expected features count)
@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status


# Predictions endpoint - The actual serving endpoint of the
# ML model
@app.post("/predict")
def predict_batch(data: List[dict]):
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(df, model_path=MODEL_PATH)

    resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp


# Providing the preview of the recently outputted predictions
@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }
