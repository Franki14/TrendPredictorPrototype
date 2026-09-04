from pathlib import Path
import pandas as pd

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "soc_media.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded!")
print(f"Original Shape: {df.shape}")

# -------------------------------------------------
# Remove Unnecessary Columns
# -------------------------------------------------

df = df.drop(columns=["Unnamed: 0", "Index"])

print(f"Shape after removing unused columns: {df.shape}")

# -------------------------------------------------
# Remove Duplicate Posts
# -------------------------------------------------

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(f"Removed {before-after} duplicate rows.")

# -------------------------------------------------
# Split Hashtags
# -------------------------------------------------

df["Hashtag"] = df["Hashtags"].str.split()

# Expand one row into multiple rows
df = df.explode("Hashtag")

print(f"Shape after exploding hashtags: {df.shape}")

# -------------------------------------------------
# Keep Useful Columns
# -------------------------------------------------

df = df[
    [
        "Text",
        "Sentiment",
        "Timestamp",
        "User",
        "Platform",
        "Hashtag",
        "Retweets",
        "Likes",
        "Year",
        "Month",
        "Day",
        "Hour",
    ]
]

print("\nFirst 10 Rows")

print(df.head(10))

print("\nNumber of Unique Hashtags")

print(df["Hashtag"].nunique())

# -------------------------------------------------
# Save cleaned data
# -------------------------------------------------

OUTPUT_PATH = PROJECT_ROOT / "data" / "clean_data.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nClean dataset saved!")