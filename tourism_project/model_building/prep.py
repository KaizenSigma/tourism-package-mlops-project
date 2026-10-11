import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]
TARGET = "ProdTaken"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill categorical missing values with mode and numerical with median
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(exclude=["object"]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df)

    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain shape: {Xtrain.shape}")
    print(f"Xtest shape: {Xtest.shape}")


if __name__ == "__main__":
    main()
