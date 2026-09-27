"""
Prediction Pipeline Verification

Verifies that the reusable prediction pipeline produces exactly
the same probabilities and classifications as direct inference
from the saved Random Forest model.

This is an engineering verification step, not model evaluation.
"""

import joblib
import pandas as pd

from predict import (
    load_metadata,
    predict_trend,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/final_random_forest.joblib"

DATA_PATH = "data/processed/cascade_validation_network.csv"

NUM_TEST_ROWS = 20

TOLERANCE = 1e-12


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "#" * 80
    )

    print(
        "PREDICTION PIPELINE VERIFICATION"
    )

    print(
        "#" * 80
    )

    # --------------------------------------------------------
    # Load metadata and model
    # --------------------------------------------------------

    metadata = load_metadata()

    features = metadata["features"]

    threshold = float(
        metadata["classification_threshold"]
    )

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Load verification data
    # --------------------------------------------------------

    print(
        "\nLoading verification dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    sample_df = df[
        features
    ].head(
        NUM_TEST_ROWS
    )

    print(
        f"Rows checked: {len(sample_df)}"
    )

    # --------------------------------------------------------
    # Verify predictions
    # --------------------------------------------------------

    probability_matches = 0
    prediction_matches = 0

    max_probability_difference = 0.0

    print(
        "\nComparing direct model inference "
        "with predict_trend()..."
    )

    for row_number, (_, row) in enumerate(
        sample_df.iterrows(),
        start=1,
    ):

        # ====================================================
        # Direct model prediction
        # ====================================================

        direct_input = pd.DataFrame(
            [row.values],
            columns=features,
        )

        direct_probability = float(
            model.predict_proba(
                direct_input
            )[0, 1]
        )

        direct_prediction = int(
            direct_probability >= threshold
        )

        # ====================================================
        # Prediction pipeline
        # ====================================================

        feature_values = {
            feature: row[feature]
            for feature in features
        }

        pipeline_result = predict_trend(
            feature_values,
            model=model,
            metadata=metadata,
        )

        pipeline_probability = (
            pipeline_result["probability"]
        )

        pipeline_prediction = (
            pipeline_result["prediction"]
        )

        # ====================================================
        # Compare
        # ====================================================

        probability_difference = abs(
            direct_probability
            - pipeline_probability
        )

        max_probability_difference = max(
            max_probability_difference,
            probability_difference,
        )

        probability_match = (
            probability_difference
            <= TOLERANCE
        )

        prediction_match = (
            direct_prediction
            == pipeline_prediction
        )

        if probability_match:
            probability_matches += 1

        if prediction_match:
            prediction_matches += 1

        print(
            f"Row {row_number:02d} | "
            f"Direct={direct_probability:.6f} | "
            f"Pipeline={pipeline_probability:.6f} | "
            f"Prediction={pipeline_prediction} | "
            f"Match={'YES' if probability_match and prediction_match else 'NO'}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "VERIFICATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Rows tested:              {len(sample_df)}"
    )

    print(
        f"Probability matches:      "
        f"{probability_matches}/{len(sample_df)}"
    )

    print(
        f"Classification matches:   "
        f"{prediction_matches}/{len(sample_df)}"
    )

    print(
        f"Maximum probability diff: "
        f"{max_probability_difference:.12f}"
    )

    # --------------------------------------------------------
    # Final assertion
    # --------------------------------------------------------

    if probability_matches != len(sample_df):
        raise AssertionError(
            "Prediction probabilities do not match."
        )

    if prediction_matches != len(sample_df):
        raise AssertionError(
            "Predicted classes do not match."
        )

    print(
        "\nPrediction pipeline verification: PASSED"
    )


if __name__ == "__main__":
    main()