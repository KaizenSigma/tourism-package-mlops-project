import pandas as pd

EXPECTED_COLUMNS = [
    "Unnamed: 0", "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

DATA_PATH = "tourism_project/data/tourism.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print("Dataset registration successful")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("")
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(dropna=False).to_string())
    print("")
    print("Missing values by column:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    main()
