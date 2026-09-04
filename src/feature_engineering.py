from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "clean_data.csv"

df = pd.read_csv(DATA_PATH)

# print(df.head())
# print(df.columns)

# -------------------------------------------------
# Engagement
# -------------------------------------------------

df["Engagement"] = df["Likes"] + df["Retweets"]

print(df[["Hashtag","Engagement"]].head())

# -------------------------------------------------
# Sentiments
# -------------------------------------------------

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# Calculate sentiment for every post
df["SentimentScore"] = df["Text"].apply(
    lambda text: sia.polarity_scores(str(text))["compound"]
)

print(df[["Text", "SentimentScore"]].head())

# -------------------------------------------------
# Network Feature
# -------------------------------------------------

network = (
    df.groupby("Hashtag")["User"]
      .nunique()
      .reset_index()
)

network.rename(
    columns={
        "User":"UniqueUsers"
    },
    inplace=True
)

# -------------------------------------------------
# Temporal
# -------------------------------------------------

temporal = (
    df.groupby("Hashtag")
      .size()
      .reset_index(name="PostCount")
)

# -------------------------------------------------
# Aggregate
# -------------------------------------------------

summary = (
    df.groupby("Hashtag")
      .agg({

          "Engagement":"mean",

          "SentimentScore":"mean"

      })
      .reset_index()
)

summary = summary.merge(network,on="Hashtag")
summary = summary.merge(temporal,on="Hashtag")
summary = summary[summary["PostCount"] >= 2]
print(f"Number of hashtags after filtering: {len(summary)}")

print("\nFeature Summary")
print(summary.head(10))

# -------------------------------------------------
# Normalize
# -------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

features = [
    "Engagement",
    "SentimentScore",
    "UniqueUsers",
    "PostCount"
]

scaler = MinMaxScaler()

summary[[f"{c}_Norm" for c in features]] = scaler.fit_transform(
    summary[features]
)


# -------------------------------------------------
# Viral Potential Score
# -------------------------------------------------

summary["VPS"] = (
      0.40 * summary["Engagement_Norm"]
    + 0.30 * summary["SentimentScore_Norm"]
    + 0.20 * summary["UniqueUsers_Norm"]
    + 0.10 * summary["PostCount_Norm"]
)

summary = summary.sort_values(
    "VPS",
    ascending=False
)

print("\nTop 10 Viral Hashtags")

print(
    summary[
        [
            "Hashtag",
            "VPS",
            "Engagement",
            "SentimentScore",
            "UniqueUsers",
            "PostCount"
        ]
    ].head(10)
)

# -------------------------------------------------
# Dashboard content display
# -------------------------------------------------

dashboard_df = summary[
    [
        "Hashtag",
        "VPS",
        "Engagement",
        "SentimentScore",
        "UniqueUsers",
        "PostCount"
    ]
]

dashboard_df.to_csv(
    PROJECT_ROOT / "data" / "viral_scores.csv",
    index=False
)

print("\nSaved viral_scores.csv")