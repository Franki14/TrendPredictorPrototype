from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_PATH = (
    PROCESSED_DIR
    / "cascade_train_network.csv"
)


# ============================================================
# Feature groups
# ============================================================

TEMPORAL_FEATURES = [
    "TimeTo5",
    "MeanInterarrival",
    "StdInterarrival",
    "MinInterarrival",
    "MaxInterarrival",
    "EarlyVelocity",
    "EarlyAcceleration",
]


NETWORK_FEATURES = [
    "MeanDegree",
    "MaxDegree",
    "DegreeStd",
    "EarlyInternalEdges",
    "EarlyDensity",
    "NeighbourhoodReach",
    "MeanClustering",
    "EarlyCommunityCount",
    "CommunityDiversity",
    "MeanPageRank",
    "MaxPageRank",
]


ALL_FEATURES = (
    TEMPORAL_FEATURES
    + NETWORK_FEATURES
)

# ============================================================
# Selected features for predictive modelling
# ============================================================

# Reduced temporal feature set.
#
# TimeTo5:
#   Main measure of early diffusion speed.
#
# MinInterarrival:
#   Captures the shortest early adoption interval and is not
#   excessively correlated with TimeTo5.
#
# EarlyAcceleration:
#   Captures changes in early diffusion timing and provides
#   information distinct from overall diffusion speed.

TEMPORAL_SELECTED = [
    "TimeTo5",
    "MinInterarrival",
    "EarlyAcceleration",
]


# Reduced network feature set.
#
# NeighbourhoodReach:
#   Represents the combined immediate social reach of the
#   first five adopters.
#
# EarlyDensity:
#   Represents cohesion among the first five adopters.
#
# MeanClustering:
#   Represents local network embeddedness.
#
# CommunityDiversity:
#   Represents how the first five adopters are distributed
#   across Louvain communities.
#
# MeanPageRank:
#   Represents average structural influence of the first
#   five adopters.

NETWORK_SELECTED = [
    "NeighbourhoodReach",
    "EarlyDensity",
    "MeanClustering",
    "CommunityDiversity",
    "MeanPageRank",
]


SELECTED_FEATURES = (
    TEMPORAL_SELECTED
    + NETWORK_SELECTED
)

# ============================================================
# Features excluded from the reduced modelling set
# ============================================================

EXCLUDED_FEATURES = {
    "MeanInterarrival":
        "Exact linear transformation of TimeTo5.",

    "StdInterarrival":
        "Highly correlated with TimeTo5.",

    "MaxInterarrival":
        "Highly correlated with TimeTo5.",

    "EarlyVelocity":
        (
            "Deterministically related to TimeTo5 for "
            "positive-duration cascades and problematic for "
            "zero-duration cascades."
        ),

    "MeanDegree":
        (
            "Highly correlated with other connectivity "
            "features; NeighbourhoodReach retained as the "
            "representative connectivity feature."
        ),

    "MaxDegree":
        (
            "Highly correlated with other connectivity "
            "features."
        ),

    "DegreeStd":
        (
            "Highly correlated with other connectivity "
            "features."
        ),

    "EarlyInternalEdges":
        (
            "Exact transformation of EarlyDensity when "
            "the observation size is fixed at five adopters."
        ),

    "EarlyCommunityCount":
        (
            "Near-perfectly correlated with "
            "CommunityDiversity."
        ),

    "MaxPageRank":
        (
            "Highly correlated with MeanPageRank; "
            "MeanPageRank retained as the group-level "
            "structural influence measure."
        ),
}

# ============================================================
# Preprocessing groups
# ============================================================

# Strongly right-skewed non-negative features that will use
# log1p transformation inside the modelling pipeline.

LOG1P_FEATURES = [
    "TimeTo5",
    "MinInterarrival",
    "NeighbourhoodReach",
]


# Features retained without log transformation.
#
# Logistic Regression will later standardize continuous
# predictors inside a scikit-learn pipeline.

NON_LOG_FEATURES = [
    "EarlyAcceleration",
    "EarlyDensity",
    "MeanClustering",
    "CommunityDiversity",
    "MeanPageRank",
]

# ============================================================
# Load training data
# ============================================================

def load_training_data():

    df = pd.read_csv(
        TRAIN_PATH
    )

    missing = [
        feature
        for feature in ALL_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing expected features: {missing}"
        )

    return df


# ============================================================
# Basic feature audit
# ============================================================

def feature_audit(df):

    rows = []

    for feature in ALL_FEATURES:

        series = df[feature]

        rows.append(
            {
                "Feature": feature,
                "Mean": series.mean(),
                "Median": series.median(),
                "Std": series.std(),
                "Min": series.min(),
                "Max": series.max(),
                "Skewness": series.skew(),
                "UniqueValues": (
                    series.nunique()
                ),
                "MissingValues": (
                    series.isna().sum()
                ),
            }
        )

    audit_df = pd.DataFrame(
        rows
    )

    return audit_df


# ============================================================
# Correlation analysis
# ============================================================

def correlation_analysis(df):

    correlations = (
        df[ALL_FEATURES]
        .corr(
            method="spearman"
        )
    )

    return correlations


# ============================================================
# Find highly correlated feature pairs
# ============================================================

def find_high_correlations(
    correlation_matrix,
    threshold=0.80,
):

    pairs = []

    for i in range(
        len(ALL_FEATURES)
    ):

        for j in range(
            i + 1,
            len(ALL_FEATURES)
        ):

            feature_a = (
                ALL_FEATURES[i]
            )

            feature_b = (
                ALL_FEATURES[j]
            )

            correlation = (
                correlation_matrix.loc[
                    feature_a,
                    feature_b,
                ]
            )

            if (
                abs(correlation)
                >= threshold
            ):

                pairs.append(
                    {
                        "Feature_A":
                            feature_a,

                        "Feature_B":
                            feature_b,

                        "Spearman":
                            correlation,

                        "AbsoluteCorrelation":
                            abs(correlation),
                    }
                )

    pairs_df = pd.DataFrame(
        pairs
    )

    if not pairs_df.empty:

        pairs_df = (
            pairs_df.sort_values(
                "AbsoluteCorrelation",
                ascending=False,
            )
        )

    return pairs_df


# ============================================================
# Check known deterministic relationships
# ============================================================

def deterministic_checks(df):

    print()
    print("=" * 80)
    print(
        "DETERMINISTIC / MATHEMATICAL RELATIONSHIP CHECKS"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # MeanInterarrival = TimeTo5 / 4
    # --------------------------------------------------------

    mean_gap_expected = (
        df["TimeTo5"] / 4
    )

    mean_gap_match = (
        np.allclose(
            df["MeanInterarrival"],
            mean_gap_expected,
        )
    )

    print(
        "MeanInterarrival == TimeTo5 / 4:",
        mean_gap_match,
    )

    # --------------------------------------------------------
    # EarlyVelocity = 4 / TimeTo5
    # --------------------------------------------------------

    # Protect against zero duration.
    nonzero = (
        df["TimeTo5"] > 0
    )

    velocity_expected = (
        4
        / df.loc[
            nonzero,
            "TimeTo5",
        ]
    )

    velocity_match = (
        np.allclose(
            df.loc[
                nonzero,
                "EarlyVelocity",
            ],
            velocity_expected,
        )
    )

    print(
        "EarlyVelocity == 4 / TimeTo5 "
        "(where TimeTo5 > 0):",
        velocity_match,
    )

    # --------------------------------------------------------
    # EarlyDensity = EarlyInternalEdges / 10
    # --------------------------------------------------------

    density_expected = (
        df["EarlyInternalEdges"]
        / 10
    )

    density_match = (
        np.allclose(
            df["EarlyDensity"],
            density_expected,
        )
    )

    print(
        "EarlyDensity == "
        "EarlyInternalEdges / 10:",
        density_match,
    )


def print_selection_summary():

    print()
    print("=" * 80)
    print("FINAL REDUCED FEATURE SET")
    print("=" * 80)

    print(
        f"\nOriginal candidate features: "
        f"{len(ALL_FEATURES)}"
    )

    print(
        f"Selected features: "
        f"{len(SELECTED_FEATURES)}"
    )

    print(
        f"Excluded features: "
        f"{len(EXCLUDED_FEATURES)}"
    )

    print(
        "\nSelected temporal features:"
    )

    for feature in TEMPORAL_SELECTED:
        print(
            f"  - {feature}"
        )

    print(
        "\nSelected network features:"
    )

    for feature in NETWORK_SELECTED:
        print(
            f"  - {feature}"
        )

    print(
        "\nExcluded features and reasons:"
    )

    for feature, reason in EXCLUDED_FEATURES.items():

        print(
            f"  - {feature}: {reason}"
        )

def validate_selected_features(df):

    missing = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Selected features missing from dataset: "
            f"{missing}"
        )

    overlap = (
        set(SELECTED_FEATURES)
        & set(EXCLUDED_FEATURES)
    )

    if overlap:

        raise ValueError(
            f"Features cannot be both selected and excluded: "
            f"{sorted(overlap)}"
        )

    if (
        set(LOG1P_FEATURES)
        | set(NON_LOG_FEATURES)
    ) != set(SELECTED_FEATURES):

        raise ValueError(
            "Preprocessing groups do not exactly match "
            "the selected feature set."
        )

    print(
        "\nSelected feature validation: PASSED"
    )

# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading training feature dataset..."
    )

    df = (
        load_training_data()
    )

    print(
        f"Training shape: "
        f"{df.shape}"
    )

    print(
        f"Candidate features: "
        f"{len(ALL_FEATURES)}"
    )

    print()
    print("=" * 80)
    print(
        "FEATURE AUDIT"
    )
    print("=" * 80)

    audit_df = (
        feature_audit(df)
    )

    print(
        audit_df
        .round(6)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "SPEARMAN FEATURE CORRELATIONS"
    )
    print("=" * 80)

    correlation_matrix = (
        correlation_analysis(df)
    )

    print(
        correlation_matrix
        .round(3)
        .to_string()
    )

    print()
    print("=" * 80)
    print(
        "HIGHLY CORRELATED PAIRS "
        "(|SPEARMAN| >= 0.80)"
    )
    print("=" * 80)

    high_correlations = (
        find_high_correlations(
            correlation_matrix,
            threshold=0.80,
        )
    )

    if high_correlations.empty:

        print(
            "No highly correlated pairs."
        )

    else:

        print(
            high_correlations[
                [
                    "Feature_A",
                    "Feature_B",
                    "Spearman",
                ]
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    deterministic_checks(
        df
    )
    print()
    print("=" * 80)
    print("ZERO-DURATION TEMPORAL CHECK")
    print("=" * 80)

    zero_time = df[
        "TimeTo5"
    ] == 0

    print(
        "Cascades with TimeTo5 == 0:",
        zero_time.sum(),
    )

    print(
        "Percentage:",
        round(
            zero_time.mean() * 100,
            3,
        ),
        "%",
    )

    print(
        "\nEarlyVelocity for zero-duration cascades:"
    )

    print(
        df.loc[
            zero_time,
            "EarlyVelocity",
        ]
        .value_counts()
        .sort_index()
    )

    validate_selected_features(
        df
    )

    print_selection_summary()

if __name__ == "__main__":
    main()