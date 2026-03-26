import sys
import os
from pathlib import Path
import pandas as pd
import pytest
from src.inference_pipeline.inference import predict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@pytest.fixture(scope="session")
def sample_df():
    """Loading a small sample of data from the
    cleaning_eval.csv for inference testing."""
    sample_path = ROOT / "data/processed/feature_engineered_eval.csv"
    df = pd.read_csv(sample_path).sample(5,
                        random_state=39).reset_index(drop=True)
    return df

def test_inference_runs_and_returns_predictions(sample_df):
    """Ensuring that the inference pipeline is running
     and is returning predicted_price column."""
    preds_df = predict(sample_df)

    # Making sure output isn't empty
    assert not preds_df.empty

    # Must include prediction column
    assert "predicted_price" in preds_df.columns

    # Predictions should be numeric
    assert (pd.api.types.is_numeric_dtype
            (preds_df["predicted_price"]))

    print("Test for the inference pipeline passed. The"
          " Predictions are:")
    print(preds_df[["predicted_price"]].head())
