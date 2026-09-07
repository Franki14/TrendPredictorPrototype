from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nltk.sentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "clean_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "viral_scores.csv"


# ---------------------------------------------------------
# 2. Load preprocessed data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"Loaded dataset: {df.shape}")


# ---------------------------------------------------------
# 3. Calculate VADER sentiment
# ---------------------------------------------------------

sia = SentimentIntensityAnalyzer()

df["SentimentScore"] = df["Text"].apply(
    lambda text: sia.polarity_scores(str(text))["compound"]
)


# ---------------------------------------------------------
# 4. Prepare hashtag data
# ---------------------------------------------------------

# Hashtags have already been extracted and exploded
# during preprocessing.
df["Hashtag"] = df["Hashtag"].astype(str).str.strip()

df = df[df["Hashtag"] != ""]

print(f"Hashtag-level dataset: {df.shape}")


# ---------------------------------------------------------
# 5. Calculate engagement
# ---------------------------------------------------------

df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0)
df["Retweets"] = pd.to_numeric(df["Retweets"], errors="coerce").fillna(0)

df["Engagement"] = df["Likes"] + df["Retweets"]


# ---------------------------------------------------------
# 6. Aggregate features by hashtag
# ---------------------------------------------------------

summary = (
    df.groupby("Hashtag")
    .agg(
        Engagement=("Engagement", "sum"),
        SentimentScore=("SentimentScore", "mean"),
        UniqueUsers=("User", "nunique"),
        PostCount=("Hashtag", "size")
    )
    .reset_index()
)


# ---------------------------------------------------------
# 7. Normalise feature values
# ---------------------------------------------------------

features = [
    "Engagement",
    "SentimentScore",
    "UniqueUsers",
    "PostCount"
]

scaler = MinMaxScaler()

normalised_features = [f"{feature}_Norm" for feature in features]

summary[normalised_features] = scaler.fit_transform(
    summary[features]
)


# ---------------------------------------------------------
# 8. Calculate Viral Potential Score
# ---------------------------------------------------------

summary["VPS"] = (
      0.40 * summary["Engagement_Norm"]
    + 0.30 * summary["SentimentScore_Norm"]
    + 0.20 * summary["UniqueUsers_Norm"]
    + 0.10 * summary["PostCount_Norm"]
)


# ---------------------------------------------------------
# 9. Sort by VPS
# ---------------------------------------------------------

summary = summary.sort_values(
    "VPS",
    ascending=False
).reset_index(drop=True)


# ---------------------------------------------------------
# 10. Validation checks
# ---------------------------------------------------------

assert summary["Hashtag"].notna().all()
assert summary[normalised_features].min().min() >= 0
assert summary[normalised_features].max().max() <= 1
assert summary["VPS"].between(0, 1).all()

print("\nFeature Summary:")
print(summary.head(10))

print("\nVPS Statistics:")
print(summary["VPS"].describe())


# ---------------------------------------------------------
# 11. Save feature-engineered dataset
# ---------------------------------------------------------

summary.to_csv(OUTPUT_PATH, index=False)

print(f"\nFeature-engineered data saved to: {OUTPUT_PATH}")