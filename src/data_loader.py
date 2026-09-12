from pathlib import Path

import pandas as pd
import numpy as np

RAW_PATH = Path("~/Projekty/RAG/data/raw/Reviews.csv")
CLEAN_PATH = Path("~/Projekty/RAG/data/processed/reviews_clean.csv")

def data_info(df):
    print(df.info())
    print(df.isnull().sum())
    print(df.duplicated(subset=["Text"]).sum())
    if "text_length" in df.columns:
        print(df["text_length"].describe())

def normalize_number(number):
    if pd.isna(number):
        return None
    number = int(number)
    return number

def clean_text(series):
    return (
        series
        .str.replace(r"<.*?>", "", regex=True)      # tagi HTML
        .str.replace(r"http\S+", "", regex=True)    # linki
        .str.strip()
    )

def filter_invalid_reviews(df, field, min_length=20, max_length=2000):
    lengths = df[field].str.len()
    df = df[(lengths > min_length) & (lengths < max_length)]
    df = df[~df[field].str.fullmatch(r"[\W\d_]+")]
    return df

def load_data():
    df = pd.read_csv(RAW_PATH)
    data_info(df)

    df["Score"] = df["Score"].apply(normalize_number)
    df = df.dropna(subset=["Text"])
    df["Text"] = clean_text(df["Text"])
    df = df.drop_duplicates(subset=["Text"])
    df = filter_invalid_reviews(df, "Text").copy()

    df["text_length"] = df["Text"].str.len()
    df["helpfulness_ratio"] = np.where(df["HelpfulnessDenominator"] > 0, df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"], np.nan)
    df["date"] = pd.to_datetime(df["Time"], unit="s")

    df_clean = df[["ProductId", "Score", "Summary", "Text", "date", "helpfulness_ratio", "text_length"]]
    data_info(df_clean)

    df_sample = df_clean.groupby("Score", group_keys=False).sample(
        frac=3000 / len(df_clean), random_state=42
    )
    
    data_info(df_sample)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(CLEAN_PATH, index=False)