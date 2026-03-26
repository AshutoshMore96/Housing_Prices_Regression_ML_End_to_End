""" Load & split the dataset using time-series as intervals """

""" Production code by default writes to the directory data/raw """

"""Followed by tests can pass a temporary 'output_dir' so that nothing 
in the directory 'data/' is tampered with """

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_and_split_data(
        raw_path: str = "data/raw/untouched_raw_original.csv",
        output_dir: Path | str = DATA_DIR,
):
    """ Load the raw datasets and further split them into
    training, evaluation, and testing data sets according to date, and
    then save to 'output_dir' folder """

    df = pd.read_csv(raw_path)

    # Make sure of the datetime and sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    #  Divide the timeline for the evaluation and testing sets
    #  accordingly to avoid data leakage
    cutoff_date_eval = pd.Timestamp("2020-01-01")       # Evaluation set begins
    cutoff_date_test = pd.Timestamp("2022-01-01")       # Evaluation set begins

    # Splitting
    train_df = df[df["date"] < cutoff_date_eval]
    eval_df = df[(df["date"] >= cutoff_date_eval)
                 &
                 (df["date"] < cutoff_date_test)]
    test_df = df[df["date"] >= cutoff_date_test]

    # Save the datasets into memory
    outdir = Path(output_dir)
    outdir.mkdir(parents = True, exist_ok = True)
    train_df.to_csv(outdir / "train.csv", index= False)
    eval_df.to_csv(outdir / "eval.csv", index= False)
    test_df.to_csv(outdir / "test.csv", index= False)

    print(f"Data splitting completed and files saved to {outdir}")
    print(f" Training: {train_df.shape}, "
          f"Evaluation: {eval_df.shape},"
          f"Testing: {test_df.shape}")
    return train_df, eval_df, test_df


if __name__ == "__main__":
    load_and_split_data()
