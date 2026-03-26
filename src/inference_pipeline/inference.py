from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

""" Inference pipeline"""
""" Taking RAW input data (same schema as the testing set)
Applying the preprocessing and feature engineering using the saved
encoders
Aligning features as per the training set
Outputting predictions """

# Raw data preprocessed then feature engineered and aligned as
# per schema used to predict the predictions

# Importing the preprocessing and feature engineering
# helping functions
from src.feature_pipeline.preprocess import (clean_and_merge,
                                             drop_duplicates, remove_outliers)
from src.feature_pipeline.feature_engineering import (add_date_features,
                                                      drop_unused_columns)

# Setting up default paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "Users "/ "ashutosh "/ "PycharmProjects" / "Housing_Prices_Regression_ML_End_to_End" / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("Inference from the projects root directory:", PROJECT_ROOT)

# Loading the feature columns for training (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]  # excluding price column
else:
    TRAIN_FEATURE_COLUMNS = None


# Inference Function

def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:

    # First Step: Preprocessing the raw input
    df = clean_and_merge(input_df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    # Second Step: Feature engineering
    if "date" in df.columns:
        df = add_date_features(df)

    # Third Step: Creating the encodings
    # Frequency encoding the 'zipcode' column
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    # Target encoding (city_full → city_full_encoded)
    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df["city_full_encoded"] = target_encoder.transform(df["city_full"])
        df = df.drop(columns=["city_full", "city"], errors="ignore")

    # object_cols = df.select_dtypes(include=["object"]).columns
    # df = df.drop(columns=object_cols)

    # Drop leakable columns
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Fourth Step: Separate actuals if present
    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Fifth Step: Align the columns according to the training schema
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Sixth Step: Loading the model for making the predictions
    model = load(model_path)
    preds = model.predict(df)

    # Seventh Step: Build output
    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out


# Entrypoint
# Allows running inference directly from terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the inference on the new housing data (raw).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Path for saving the predictions into CSV file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path to the trained model file")
    parser.add_argument("--freq_encoder", type=str, default=str(DEFAULT_FREQ_ENCODER),
                        help="Path to frequency encoder pickle")
    parser.add_argument("--target_encoder", type=str,
                        default=str(DEFAULT_TARGET_ENCODER),
                        help="Path to target encoder pickle")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"Predictions will be saved to {args.output}")
