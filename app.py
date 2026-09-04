from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "viral_scores.csv"

df = pd.read_csv(DATA_PATH)

st.set_page_config(
    page_title="Social Media Trend Predictor",
    layout="wide"
)

st.title("📈 Social Media Trend Predictor")

st.write(
    "Feature Prototype: Viral Potential Score (VPS)"
)

st.subheader("🔥 Top 10 Trending Hashtags")

top10 = df.sort_values(
    "VPS",
    ascending=False
).head(10)

st.dataframe(top10)

# -------------------------------------------------
# Searchbox
# -------------------------------------------------

hashtag = st.selectbox(

    "Choose a hashtag",

    sorted(df["Hashtag"])

)

# -------------------------------------------------
# Metrics
# -------------------------------------------------

row = df[
    df["Hashtag"] == hashtag
].iloc[0]
#why start with 0
col1,col2,col3,col4,col5 = st.columns(5)

col1.metric(
    "VPS",
    f"{row['VPS']:.2f}"
)

col2.metric(
    "Engagement",
    f"{row['Engagement']:.2f}"
)

col3.metric(
    "Sentiment",
    f"{row['SentimentScore']:.2f}"
)

col4.metric(
    "Users",
    int(row["UniqueUsers"])
)

col5.metric(
    "Posts",
    int(row["PostCount"])
)

# -------------------------------------------------
# Bar Chart - figure out plotly
# -------------------------------------------------

st.subheader("Top Viral Potential Scores")

chart = top10.set_index("Hashtag")

st.bar_chart(chart["VPS"])

# -------------------------------------------------
# Feature Breakdown
# -------------------------------------------------

st.subheader("Feature Breakdown")

feature_df = pd.DataFrame({

    "Feature":[
        "Engagement",
        "Sentiment",
        "Network",
        "Temporal"
    ],

    "Value":[

        row["Engagement"],

        row["SentimentScore"],

        row["UniqueUsers"],

        row["PostCount"]

    ]

})

st.bar_chart(
    feature_df.set_index("Feature")
)