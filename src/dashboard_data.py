"""
Dashboard Data Layer

Provides the connection between processed cascade data,
the frozen prediction pipeline, and the dashboard.

The model receives only the eight established predictive features.
Target and FinalSize are retained only as historical/descriptive
information and are never supplied as model inputs.
"""

import pandas as pd

from predict import (
    load_model,
    load_metadata,
    predict_trend,
)


# ============================================================
# Configuration
# ============================================================

VALIDATION_PATH = (
    "data/processed/cascade_validation_network.csv"
)


# ============================================================
# Load Dashboard Data
# ============================================================

def load_dashboard_data(
    filepath=VALIDATION_PATH,
):
    """
    Load cascade records available to the prototype.
    """

    df = pd.read_csv(filepath)

    required_columns = [
        "CascadeID",
        "Split",
        "FinalSize",
        "Target",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dashboard dataset missing columns: {missing}"
        )

    if df["CascadeID"].duplicated().any():
        raise ValueError(
            "CascadeID must uniquely identify each cascade."
        )

    return df


# ============================================================
# Retrieve Cascade
# ============================================================

def get_cascade(
    df,
    cascade_id,
):
    """
    Retrieve one cascade using its CascadeID.
    """

    matches = df[
        df["CascadeID"] == cascade_id
    ]

    if matches.empty:
        raise ValueError(
            f"CascadeID {cascade_id} was not found."
        )

    if len(matches) > 1:
        raise ValueError(
            f"CascadeID {cascade_id} is not unique."
        )

    return matches.iloc[0]


# ============================================================
# Extract Model Features
# ============================================================

def extract_model_features(
    cascade,
    metadata,
):
    """
    Extract only the features expected by the frozen model.

    Target and FinalSize are deliberately excluded.
    """

    required_features = metadata[
        "features"
    ]

    missing = [
        feature
        for feature in required_features
        if feature not in cascade.index
    ]

    if missing:
        raise ValueError(
            f"Cascade missing model features: {missing}"
        )

    return {
        feature: cascade[feature]
        for feature in required_features
    }


# ============================================================
# Historical Information
# ============================================================

def get_historical_information(
    cascade,
):
    """
    Return descriptive information about the historical cascade.

    These values are NOT model inputs.
    """

    target = int(
        cascade["Target"]
    )

    return {
        "cascade_id": int(
            cascade["CascadeID"]
        ),

        "split": str(
            cascade["Split"]
        ),

        "final_size": int(
            cascade["FinalSize"]
        ),

        "actual_target": target,

        "actual_label": (
            "Emerging Trend"
            if target == 1
            else "Non-Emerging Trend"
        ),
    }


# ============================================================
# Cascade Characteristics
# ============================================================

def get_cascade_characteristics(
    cascade,
):
    """
    Return additional descriptive characteristics for the
    Trend Explorer section of the prototype.

    These are for exploration only and are not necessarily
    inputs to the final prediction model.
    """

    characteristic_columns = [
        "TimeTo5",
        "MeanInterarrival",
        "StdInterarrival",
        "MinInterarrival",
        "MaxInterarrival",
        "EarlyVelocity",
        "EarlyAcceleration",
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

    return {
        column: float(cascade[column])
        for column in characteristic_columns
        if column in cascade.index
    }


# ============================================================
# Build Dashboard Record
# ============================================================

def build_dashboard_record(
    cascade_id,
    df=None,
    model=None,
    metadata=None,
):
    """
    Build the complete information required by the dashboard
    for a selected historical cascade.
    """

    if df is None:
        df = load_dashboard_data()

    if metadata is None:
        metadata = load_metadata()

    if model is None:
        model = load_model()

    cascade = get_cascade(
        df,
        cascade_id,
    )

    model_features = extract_model_features(
        cascade,
        metadata,
    )

    prediction = predict_trend(
        model_features,
        model=model,
        metadata=metadata,
    )

    historical = get_historical_information(
        cascade
    )

    characteristics = (
        get_cascade_characteristics(
            cascade
        )
    )

    prediction_correct = (
        prediction["prediction"]
        == historical["actual_target"]
    )

    return {
        "historical": historical,
        "prediction": prediction,
        "prediction_correct": prediction_correct,
        "model_features": model_features,
        "characteristics": characteristics,
    }


# ============================================================
# Display Verification
# ============================================================

def print_dashboard_record(
    record,
):
    """
    Display a dashboard record for engineering verification.
    """

    historical = record[
        "historical"
    ]

    prediction = record[
        "prediction"
    ]

    print(
        "\n" + "=" * 75
    )

    print(
        f"CASCADE {historical['cascade_id']}"
    )

    print(
        "=" * 75
    )

    print(
        f"Dataset split:       "
        f"{historical['split']}"
    )

    print(
        f"Historical final size: "
        f"{historical['final_size']}"
    )

    print(
        f"Actual outcome:      "
        f"{historical['actual_label']}"
    )

    print(
        f"Model prediction:    "
        f"{prediction['prediction_label']}"
    )

    print(
        f"Probability:         "
        f"{prediction['probability']:.4f}"
    )

    print(
        f"VPS:                 "
        f"{prediction['vps']:.2f}/100"
    )

    print(
        f"VPS level:           "
        f"{prediction['vps_category']}"
    )

    print(
        f"Prediction correct:  "
        f"{record['prediction_correct']}"
    )

    print(
        "\nModel features:"
    )

    for feature, value in record[
        "model_features"
    ].items():

        print(
            f"  {feature}: {value}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "#" * 80
    )

    print(
        "DASHBOARD DATA LAYER"
    )

    print(
        "#" * 80
    )

    df = load_dashboard_data()

    metadata = load_metadata()

    model = load_model()

    print(
        f"\nAvailable cascades: {len(df)}"
    )

    print(
        f"CascadeID range: "
        f"{df['CascadeID'].min()}–"
        f"{df['CascadeID'].max()}"
    )

    # --------------------------------------------------------
    # Test several historical examples
    # --------------------------------------------------------

    test_ids = (
        df["CascadeID"]
        .head(3)
        .tolist()
    )

    for cascade_id in test_ids:

        record = build_dashboard_record(
            cascade_id,
            df=df,
            model=model,
            metadata=metadata,
        )

        print_dashboard_record(
            record
        )

    print(
        "\nDashboard data layer: PASSED"
    )


if __name__ == "__main__":
    main()