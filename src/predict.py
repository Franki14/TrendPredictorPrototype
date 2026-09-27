"""
Trend Prediction Pipeline

Loads the frozen Random Forest model and
provides a reusable prediction interface for the Trend Predictor
prototype.

The Viral Potential Score (VPS) is derived from the model's
predicted probability for the positive trend-emergence class.

No model training occurs in this script.
"""

import json
import os

import joblib
import pandas as pd


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/final_random_forest.joblib"
METADATA_PATH = "models/model_metadata.json"


# ============================================================
# Load Metadata
# ============================================================

def load_metadata():
    """
    Load model metadata generated
    """

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    required_keys = [
        "model_name",
        "features",
        "classification_threshold",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in metadata
    ]

    if missing_keys:
        raise ValueError(
            "Model metadata is missing required fields: "
            f"{missing_keys}"
        )

    return metadata


# ============================================================
# Load Model
# ============================================================

def load_model():
    """
    Load the frozen trained model from disk.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# Validate Input Features
# ============================================================

def validate_features(
    feature_values,
    required_features,
):
    """
    Validate a dictionary containing the model input features.
    """

    if not isinstance(feature_values, dict):
        raise TypeError(
            "Prediction input must be provided as a dictionary."
        )

    # --------------------------------------------------------
    # Check for missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in required_features
        if feature not in feature_values
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Check for unexpected features
    # --------------------------------------------------------

    unexpected_features = [
        feature
        for feature in feature_values
        if feature not in required_features
    ]

    if unexpected_features:
        raise ValueError(
            "Unexpected features supplied: "
            f"{unexpected_features}"
        )

    # --------------------------------------------------------
    # Check values
    # --------------------------------------------------------

    validated_values = {}

    for feature in required_features:

        value = feature_values[feature]

        if isinstance(value, bool):
            raise TypeError(
                f"{feature} must be numeric."
            )

        try:
            numeric_value = float(value)

        except (TypeError, ValueError):
            raise TypeError(
                f"{feature} must be numeric. "
                f"Received: {value!r}"
            )

        if not pd.notna(numeric_value):
            raise ValueError(
                f"{feature} cannot contain a missing value."
            )

        validated_values[feature] = numeric_value

    return validated_values


# ============================================================
# Prepare Model Input
# ============================================================

def prepare_input(
    feature_values,
    required_features,
):
    """
    Convert validated feature values into a one-row DataFrame
    using exactly the feature order expected by the model.
    """

    validated_values = validate_features(
        feature_values,
        required_features,
    )

    input_df = pd.DataFrame(
        [
            [
                validated_values[feature]
                for feature in required_features
            ]
        ],
        columns=required_features,
    )

    return input_df


# ============================================================
# VPS Interpretation
# ============================================================

def interpret_vps(vps):
    """
    Convert the continuous VPS into a human-readable category.

    These categories are presentation bands only. They do not
    represent additional trained decision thresholds.
    """

    if vps < 25:
        return "Low"

    if vps < 50:
        return "Moderate"

    if vps < 75:
        return "High"

    return "Very High"


# ============================================================
# Prediction
# ============================================================

def predict_trend(
    feature_values,
    model=None,
    metadata=None,
):
    """
    Predict trend emergence from the eight established features.

    Returns:
        prediction
        probability
        VPS
        VPS category
        prediction label
    """

    if metadata is None:
        metadata = load_metadata()

    if model is None:
        model = load_model()

    required_features = metadata["features"]

    threshold = float(
        metadata["classification_threshold"]
    )

    input_df = prepare_input(
        feature_values,
        required_features,
    )

    # --------------------------------------------------------
    # Predict positive-class probability
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    # --------------------------------------------------------
    # Apply established classification threshold
    # --------------------------------------------------------

    prediction = int(
        probability >= threshold
    )

    # --------------------------------------------------------
    # Viral Potential Score
    # --------------------------------------------------------

    vps = probability * 100

    # --------------------------------------------------------
    # Human-readable labels
    # --------------------------------------------------------

    prediction_label = (
        "Emerging Trend"
        if prediction == 1
        else "Non-Emerging Trend"
    )

    vps_category = interpret_vps(
        vps
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probability": probability,
        "probability_percent": probability * 100,
        "vps": vps,
        "vps_category": vps_category,
        "classification_threshold": threshold,
    }

    return result


# ============================================================
# Display Prediction
# ============================================================

def print_prediction(result):
    """
    Print prediction result in a readable format.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "TREND PREDICTION"
    )

    print(
        "=" * 70
    )

    print(
        f"Prediction:  {result['prediction_label']}"
    )

    print(
        f"Probability: {result['probability']:.4f}"
    )

    print(
        f"VPS:         {result['vps']:.2f}/100"
    )

    print(
        f"VPS Level:   {result['vps_category']}"
    )

    print(
        f"Threshold:   {result['classification_threshold']:.2f}"
    )


# ============================================================
# Demonstration
# ============================================================

def main():

    print(
        "\n" + "#" * 70
    )

    print(
        "TREND PREDICTION PIPELINE"
    )

    print(
        "#" * 70
    )

    metadata = load_metadata()
    model = load_model()

    print(
        f"\nLoaded model: {metadata['model_name']}"
    )

    print(
        f"Required features: {len(metadata['features'])}"
    )

    # --------------------------------------------------------
    # Demonstration input
    #
    # This example is only used to verify that the inference
    # pipeline works. It is not used for model evaluation.
    # --------------------------------------------------------

    example_features = {
        "TimeTo5": 1000,
        "MinInterarrival": 100,
        "EarlyAcceleration": 5000,
        "NeighbourhoodReach": 100,
        "EarlyDensity": 0.20,
        "MeanClustering": 0.10,
        "CommunityDiversity": 3,
        "MeanPageRank": 0.01,
    }

    print(
        "\nExample input:"
    )

    for feature, value in example_features.items():
        print(
            f"{feature}: {value}"
        )

    result = predict_trend(
        example_features,
        model=model,
        metadata=metadata,
    )

    print_prediction(
        result
    )

    print(
        "\nPrediction pipeline: PASSED"
    )


if __name__ == "__main__":
    main()