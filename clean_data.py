import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_data(df):
    # Drop difficulty
    df.drop(columns=["difficulty"], inplace=True)

    # Missing values
    print(f"Missing Values: {df.isnull().sum()}")

    # Duplicates
    print(f"Duplicates: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)

    # Handles categorical columns, into int
    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

    # Remake into binary: normal vs attack
    df["label_binary"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Scale numerical columns, works better for ML algo
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols = [c for c in num_cols if "label" not in c]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df