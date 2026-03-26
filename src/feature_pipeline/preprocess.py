""" Read into memory the training, evaluation and
testing sets as csv files from the 'data/raw' directory"""
from os import mkdir
from unittest.mock import inplace

from scipy.signal import dfreqresp

""" Clean and then normalize the city names"""

""" Mapping the city names to the metro names and 
further merge with latitudes and longitudes"""

""" Remove the duplicates and the extreme outliers"""

""" Save the cleaned and processes sets into memory at the
directory 'data/processed'"""

""" Production will default read from 'data/raw' and write 
to directory 'data/procesed'"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents = True, exist_ok = True)

# Fixing the known mismatches manually to bring it to
# normalized form

CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}


def normalize_city(s: str) -> str:
    """ Lowercase, strip, unify dashes. Safe for processing
    'NA' values"""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = re.sub(r"[---]","-", s)     # unify dashes
    s = re.sub(r"\s+"," ", s)       # collapse spaces
    return s


def clean_and_merge(df: pd.DataFrame,
                    metros_path: str | None =
                    "data/raw/usmetros.csv") -> pd.DataFrame:
    """ Normaliize the names of cities

Merge latitudes and longitudes from the metros dataset
into training, evaluation and test dataset

If there are missing columns such as 'city_full' or
'metros_path', skip them """

    if "city_full" not in df.columns:
        print("Skipping the city merge: No 'city_full' column present")
        return df

    # Normalize the 'city_full' column
    df["city_full"] = df["city_full"].apply(normalize_city)

    #Apply mapping
    norm_mapping = {normalize_city(k): normalize_city(v)
                    for k, v in CITY_MAPPING.items()}
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # If latitude and longitude already present, skip merge
    if {"lat", "lng"}.issubset(df.columns):
        print("Skipping the latitudes and longitudes merging"
              " as they are already present in Dataframe")
        return df

    # If no metros file is provided or exists, skip the merge
    if not metros_path or not Path(metros_path).exists():
        print("Skipping the latitudes and longitudes merging as "
              "there is no file provided or found")
        return df

    # Merging of the latitudes and longitudes
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(metros.columns):
        print("Skipping the merging of latitudes and longitudes"
          " as the metros file is missing required columns")
        return df

    metros["metro_full"] = metros["metro_full"].apply(normalize_city)
    df = df.merge(metros[["metro_full", "lat", "lng"]],
                  how = "left", left_on = "city_full",
                  right_on = "metro_full")
    df.drop(columns = ["metro_full"], inplace = True,
            errors = "ignore")

    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("Still missing the latitudes and longitudes for:",
              missing)
    else:
        print("All cities matched with metros dataset")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the exact duplicates while keeping
    the different dates and years"""
    before = df.shape[0]
    df = df.drop_duplicates(subset = df.columns.
                            difference(["date", "year"]),
                            keep = False)
    after = df.shape[0]
    print(f"Removed the {before - after} duplicate rows "
          f"(Excluding date and year")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers in
    'median_list_price' > 19M column"""
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= 19_000_000].copy()
    after = df.shape[0]
    print(f"Removed the {before - after} rows with the "
          f"'median_list_price' > 19M")
    return df


def preprocess_split( split: str,
                      raw_dir: Path | str = RAW_DIR,
                      processed_dir: Path | str = PROCESSED_DIR,
                      metros_path: str | None = "data/raw/usmetros.csv",
                      ) -> pd.DataFrame:
    """Run the preprocessing function for the split and
    save it to 'preprocessed_dir' directory"""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents = True, exist_ok = True)

    path = raw_dir/ f"{split}.csv"
    df = pd.read_csv(path)

    df = clean_and_merge(df, metros_path= metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    out_path = processed_dir / f"cleaning_{split}.csv"
    df.to_csv(out_path, index = False)
    print(f"Preprocessed {split} saved to "
          f"{out_path} ({df.shape})")
    return df


def run_preprocess(
        splits: tuple[str, ...] = ("train", "eval", "test"),
        raw_dir: Path | str = RAW_DIR,
        processed_dir: Path | str = PROCESSED_DIR,
        metros_path: str | None = "data/raw/usmetros.csv",
):
    for s in splits:
        preprocess_split(s, raw_dir = raw_dir,
                         processed_dir = processed_dir,
                         metros_path = metros_path)


if __name__ == "__main__":
    run_preprocess()
