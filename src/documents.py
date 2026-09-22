import pandas as pd
from langchain_core.documents import Document


def normalize_ratio(value):
    if pd.isna(value):
        return None
    return float(value)


def dataframe_to_documents(df):
    df = df.copy()
    df["Summary"] = df["Summary"].fillna("")
    df["date"] = df["date"].astype(str)

    documents = []
    for row in df.itertuples(index=False):
        metadata = {
            "ProductId": str(row.ProductId),
            "Score": int(row.Score),
            "Summary": str(row.Summary),
            "date": row.date,
            "helpfulness_ratio": normalize_ratio(row.helpfulness_ratio),
        }
        documents.append(Document(page_content=row.Text, metadata=metadata))

    return documents
