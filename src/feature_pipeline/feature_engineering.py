""" Feature Engineering: Date parts, frequency encoding,
target encoding, drop leakage columns"""

""" 1. Read the cleaned training and evaluation CSV files
    2. Apply feature engineering steps
     3. Save the feature engineered CSV files
    4. Also store the fitted encoders for the inference """

from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump  # joblib.dump saves encoders
# and mappings to disk (important for using at inference

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents = True, exist_ok = True)


# Feature Engineering Functions

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month

    # Keeping it after the date for improved readability
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df

# To create a frequency encoding (how often a value will appear)
# Fitting only on the training set and eventually applying on the evaluation set

def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame,
                     col: str):
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    eval[f"{col}_freq"] = eval[col].map(freq_map).fillna(0)
    return train, eval, freq_map

# Using the target encoding (replace the category with
# the average of target variable)
# Fitted only on training set ( to prevent leakage)

def target_encode(train: pd.DataFrame, eval: pd.DataFrame,
                  col: str, target: str):
    """ Using the target encoder on 'col', consistently name
    as <col>_encoded
    For 'city_full' -> 'city_ful_encoded' (keeping schema
    aligned with the inference)"""

    te = TargetEncoder(cols = [col])
    encoded_col = f"{col}_encoded" if col!= "city_full" else "city_full_encoded"
    train[encoded_col] = te.fit_transform(train[col], train[target])
    eval[encoded_col] = te.transform(eval[col])
    return train, eval, te


def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    drop_cols = ["date", "city_full", "zipcode",
                 "median_sale_price"]
    train = train.drop(columns = [c for c in drop_cols if c in
                       train.columns], errors = "ignore")
    eval = eval.drop(columns=[c for c in drop_cols if c in
                                eval.columns], errors="ignore")
    return  train, eval


# Creating the Pipeline

# Handling full pipeline
# Reads cleaned CSV's and then apply feature engineering
#Save the engineered features and the encoders

def run_feature_engineering(
        in_train_path: Path | str | None = None,
        in_eval_path: Path | str | None = None,
        in_test_path: Path | str | None = None,
        output_dir: Path | str = PROCESSED_DIR,
):
    """ Run the script of feature engineering and then write the
    outputs and encoder to the disk
    Applying the same transformations to the training,
    evaluation, and testing set"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    # Default for the inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_test_path is None:
        in_test_path = PROCESSED_DIR / "cleaning_test.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    test_df = pd.read_csv(in_test_path)

    print("Training set date range:", train_df["date"].min(),
          "to", train_df["date"].max())
    print("Evaluation set date range:", eval_df["date"].min(),
          "to", eval_df["date"].max())
    print("Testing set date range:", test_df["date"].min(),
          "to", test_df["date"].max())

    # Features for the dates
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    test_df = add_date_features(test_df)

    # Frequency encoding zipcodes (fitting on the training
    # set only to avoid data leakage)
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encode(train_df,
                                                       eval_df, "zipcode")
        test_df["zipcode_freq"] = test_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl") # Saving
        # the mapping

    # Encode the target column 'city_full' (fitting on the
    # training set only)
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encode(
            train_df, eval_df, "city_full", "price")
        test_df["city_full_encoded"] = target_encoder.transform(test_df["city_full"])
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl") #Saving
        # the encoder

    # Removing the leakage prone and raw categoricals
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    test_df, _ = drop_unused_columns(test_df.copy(), test_df.copy())


    # Save the feature engineered data
    out_train_path = output_dir / "feature_engineered_train.csv"
    out_eval_path = output_dir / "feature_engineered_eval.csv"
    out_test_path = output_dir / "feature_engineered_test.csv"
    train_df.to_csv(out_train_path, index = False)
    eval_df.to_csv(out_eval_path, index = False)
    test_df.to_csv(out_test_path, index = False)

    print("Feature Engineering Complete")
    print("Training set shape:", train_df.shape)
    print("Evaluation set shape:", eval_df.shape)
    print("Testing set shape:", test_df.shape)
    print("Encoders saved to models")

    return train_df, eval_df, test_df, freq_map, target_encoder

if __name__ == "__main__":
    run_feature_engineering()
