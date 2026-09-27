"""
End-to-End System Testing

Tests the final TrendPredictor inference system from processed
cascade data through the saved model, prediction pipeline and
dashboard data layer.

This script does NOT retrain the model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import warnings

# ============================================================
# Warning Configuration
# ============================================================

# Suppress known scikit-learn parallel configuration warning.
# This warning does not affect prediction results.
warnings.filterwarnings(
    "ignore",
    message=(
        r"`sklearn\.utils\.parallel\.delayed` should be used with "
        r"`sklearn\.utils\.parallel\.Parallel`.*"
    ),
    category=UserWarning,
)

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from predict import (
    load_model,
    load_metadata,
    predict_trend,
)

from dashboard_data import (
    load_dashboard_data,
    get_cascade,
    extract_model_features,
    build_dashboard_record,
)


# ============================================================
# Configuration
# ============================================================

EXPECTED_FEATURE_COUNT = 8

EXPECTED_FEATURES = [
    "TimeTo5",
    "MinInterarrival",
    "EarlyAcceleration",
    "NeighbourhoodReach",
    "EarlyDensity",
    "MeanClustering",
    "CommunityDiversity",
    "MeanPageRank",
]

EXPECTED_THRESHOLD = 0.50

EXPECTED_REFERENCE_RESULTS = {
    0: {
        "probability": 0.164,
        "prediction": 0,
    },
    1: {
        "probability": 0.614,
        "prediction": 1,
    },
    2: {
        "probability": 0.708,
        "prediction": 1,
    },
}

PROBABILITY_TOLERANCE = 1e-12


# ============================================================
# Test Utilities
# ============================================================

def print_test(name, passed, detail=""):

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {name}"
    )

    if detail:
        print(
            f"       {detail}"
        )

    if not passed:
        raise AssertionError(
            f"{name}: {detail}"
        )


# ============================================================
# Test 1 — Artifact Loading
# ============================================================

def test_artifact_loading():

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 1 — SAVED ARTIFACT LOADING"
    )
    print(
        "=" * 80
    )

    model = load_model()
    metadata = load_metadata()

    print_test(
        "Model loads successfully",
        model is not None,
    )

    print_test(
        "Metadata loads successfully",
        metadata is not None,
    )

    return model, metadata


# ============================================================
# Test 2 — Metadata Contract
# ============================================================

def test_metadata_contract(metadata):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 2 — MODEL METADATA CONTRACT"
    )
    print(
        "=" * 80
    )

    features = metadata.get(
        "features"
    )

    print_test(
        "Metadata contains feature list",
        features is not None,
    )

    print_test(
        "Feature count is eight",
        len(features) == EXPECTED_FEATURE_COUNT,
        f"Found {len(features)} features.",
    )

    print_test(
        "Feature order matches frozen model",
        features == EXPECTED_FEATURES,
        f"Features: {features}",
    )

    threshold = metadata.get(
        "classification_threshold",
        metadata.get(
            "threshold",
            EXPECTED_THRESHOLD,
        ),
    )

    print_test(
        "Classification threshold is 0.50",
        np.isclose(
            float(threshold),
            EXPECTED_THRESHOLD,
        ),
        f"Threshold: {threshold}",
    )


# ============================================================
# Test 3 — Dashboard Dataset
# ============================================================

def test_dashboard_dataset():

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 3 — DASHBOARD DATASET"
    )
    print(
        "=" * 80
    )

    df = load_dashboard_data()

    print_test(
        "Dashboard dataset loads",
        len(df) > 0,
        f"Rows: {len(df)}",
    )

    print_test(
        "CascadeID exists",
        "CascadeID" in df.columns,
    )

    print_test(
        "CascadeID is unique",
        not df["CascadeID"].duplicated().any(),
    )

    print_test(
        "Target exists for retrospective comparison",
        "Target" in df.columns,
    )

    print_test(
        "FinalSize exists for retrospective comparison",
        "FinalSize" in df.columns,
    )

    return df


# ============================================================
# Test 4 — No Target Leakage
# ============================================================

def test_no_target_leakage(metadata):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 4 — PREDICTIVE FEATURE LEAKAGE CHECK"
    )
    print(
        "=" * 80
    )

    features = metadata["features"]

    forbidden = {
        "Target",
        "FinalSize",
        "Split",
        "CascadeID",
    }

    leaked = [
        feature
        for feature in features
        if feature in forbidden
    ]

    print_test(
        "No retrospective/outcome columns are model inputs",
        len(leaked) == 0,
        (
            "No leakage detected."
            if not leaked
            else f"Leaked columns: {leaked}"
        ),
    )


# ============================================================
# Test 5 — Reference Cascade Reproduction
# ============================================================

def test_reference_cascades(
    df,
    model,
    metadata,
):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 5 — REFERENCE CASCADE REPRODUCTION"
    )
    print(
        "=" * 80
    )

    for cascade_id, expected in (
        EXPECTED_REFERENCE_RESULTS.items()
    ):

        record = build_dashboard_record(
            cascade_id,
            df=df,
            model=model,
            metadata=metadata,
        )

        prediction = record[
            "prediction"
        ]

        probability = float(
            prediction["probability"]
        )

        predicted_class = int(
            prediction["prediction"]
        )

        expected_probability = expected[
            "probability"
        ]

        expected_class = expected[
            "prediction"
        ]

        probability_match = np.isclose(
            probability,
            expected_probability,
            atol=PROBABILITY_TOLERANCE,
            rtol=0,
        )

        class_match = (
            predicted_class
            == expected_class
        )

        print_test(
            f"Cascade {cascade_id} probability",
            probability_match,
            (
                f"Expected {expected_probability:.6f}, "
                f"received {probability:.6f}"
            ),
        )

        print_test(
            f"Cascade {cascade_id} classification",
            class_match,
            (
                f"Expected {expected_class}, "
                f"received {predicted_class}"
            ),
        )


# ============================================================
# Test 6 — Direct vs Pipeline Inference
# ============================================================

def test_direct_vs_pipeline(
    df,
    model,
    metadata,
):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 6 — DIRECT MODEL VS PIPELINE"
    )
    print(
        "=" * 80
    )

    sample = df.head(20)

    probability_differences = []

    for _, row in sample.iterrows():

        cascade_id = int(
            row["CascadeID"]
        )

        cascade = get_cascade(
            df,
            cascade_id,
        )

        features = extract_model_features(
            cascade,
            metadata,
        )

        pipeline_result = predict_trend(
            features,
            model=model,
            metadata=metadata,
        )

        feature_frame = pd.DataFrame(
            [
                [
                    features[feature]
                    for feature in metadata[
                        "features"
                    ]
                ]
            ],
            columns=metadata["features"],
        )

        direct_probability = float(
            model.predict_proba(
                feature_frame
            )[0, 1]
        )

        pipeline_probability = float(
            pipeline_result[
                "probability"
            ]
        )

        difference = abs(
            direct_probability
            - pipeline_probability
        )

        probability_differences.append(
            difference
        )

    max_difference = max(
        probability_differences
    )

    print_test(
        "Direct and pipeline probabilities match",
        max_difference <= PROBABILITY_TOLERANCE,
        (
            f"Rows tested: {len(sample)}; "
            f"maximum difference: "
            f"{max_difference:.12f}"
        ),
    )


# ============================================================
# Test 7 — VPS Consistency
# ============================================================

def test_vps_consistency(
    df,
    model,
    metadata,
):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 7 — VPS CONSISTENCY"
    )
    print(
        "=" * 80
    )

    sample = df.head(20)

    for _, row in sample.iterrows():

        cascade_id = int(
            row["CascadeID"]
        )

        record = build_dashboard_record(
            cascade_id,
            df=df,
            model=model,
            metadata=metadata,
        )

        prediction = record[
            "prediction"
        ]

        expected_vps = (
            float(
                prediction[
                    "probability"
                ]
            )
            * 100
        )

        actual_vps = float(
            prediction["vps"]
        )

        if not np.isclose(
            expected_vps,
            actual_vps,
            atol=1e-10,
            rtol=0,
        ):

            print_test(
                f"Cascade {cascade_id} VPS",
                False,
                (
                    f"Expected {expected_vps}, "
                    f"received {actual_vps}"
                ),
            )

    print_test(
        "VPS equals probability × 100",
        True,
        f"Rows tested: {len(sample)}",
    )


# ============================================================
# Test 8 — Missing Cascade
# ============================================================

def test_missing_cascade(df):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 8 — INVALID CASCADE HANDLING"
    )
    print(
        "=" * 80
    )

    invalid_id = int(
        df["CascadeID"].max()
    ) + 1000

    try:

        get_cascade(
            df,
            invalid_id,
        )

    except ValueError:

        print_test(
            "Unknown CascadeID rejected",
            True,
            f"Tested invalid ID: {invalid_id}",
        )

        return

    print_test(
        "Unknown CascadeID rejected",
        False,
        "No ValueError was raised.",
    )


# ============================================================
# Test 9 — Missing Feature
# ============================================================

def test_missing_feature(
    df,
    metadata,
):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 9 — MISSING FEATURE HANDLING"
    )
    print(
        "=" * 80
    )

    cascade = df.iloc[0].copy()

    missing_feature = (
        metadata["features"][0]
    )

    cascade = cascade.drop(
        labels=[
            missing_feature
        ]
    )

    try:

        extract_model_features(
            cascade,
            metadata,
        )

    except ValueError:

        print_test(
            "Missing required feature rejected",
            True,
            f"Removed feature: {missing_feature}",
        )

        return

    print_test(
        "Missing required feature rejected",
        False,
        (
            f"Removing {missing_feature} "
            "did not raise ValueError."
        ),
    )


# ============================================================
# Test 10 — Probability Bounds
# ============================================================

def test_probability_bounds(
    df,
    model,
    metadata,
):

    print(
        "\n" + "=" * 80
    )
    print(
        "TEST 10 — OUTPUT BOUNDS"
    )
    print(
        "=" * 80
    )

    sample = df.head(50)

    probabilities = []
    vps_values = []

    for _, row in sample.iterrows():

        cascade_id = int(
            row["CascadeID"]
        )

        record = build_dashboard_record(
            cascade_id,
            df=df,
            model=model,
            metadata=metadata,
        )

        probabilities.append(
            float(
                record[
                    "prediction"
                ][
                    "probability"
                ]
            )
        )

        vps_values.append(
            float(
                record[
                    "prediction"
                ][
                    "vps"
                ]
            )
        )

    probabilities_valid = all(
        0 <= value <= 1
        for value in probabilities
    )

    vps_valid = all(
        0 <= value <= 100
        for value in vps_values
    )

    print_test(
        "Probabilities remain within [0, 1]",
        probabilities_valid,
        f"Rows tested: {len(sample)}",
    )

    print_test(
        "VPS remains within [0, 100]",
        vps_valid,
        f"Rows tested: {len(sample)}",
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "#" * 80
    )

    print(
        "END-TO-END SYSTEM TESTING"
    )

    print(
        "#" * 80
    )

    model, metadata = (
        test_artifact_loading()
    )

    test_metadata_contract(
        metadata
    )

    df = test_dashboard_dataset()

    test_no_target_leakage(
        metadata
    )

    test_reference_cascades(
        df,
        model,
        metadata,
    )

    test_direct_vs_pipeline(
        df,
        model,
        metadata,
    )

    test_vps_consistency(
        df,
        model,
        metadata,
    )

    test_missing_cascade(
        df
    )

    test_missing_feature(
        df,
        metadata,
    )

    test_probability_bounds(
        df,
        model,
        metadata,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "ALL END-TO-END TESTS PASSED"
    )

    print(
        "=" * 80
    )

    print(
        "\nVerified:"
    )

    print(
        "  - saved model loading"
    )

    print(
        "  - metadata/model feature contract"
    )

    print(
        "  - dashboard dataset integrity"
    )

    print(
        "  - absence of target/future-outcome leakage"
    )

    print(
        "  - reference prediction reproducibility"
    )

    print(
        "  - direct/pipeline inference equivalence"
    )

    print(
        "  - VPS calculation consistency"
    )

    print(
        "  - invalid CascadeID handling"
    )

    print(
        "  - missing-feature handling"
    )

    print(
        "  - probability and VPS bounds"
    )


if __name__ == "__main__":
    main()