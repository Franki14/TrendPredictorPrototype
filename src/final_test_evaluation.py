import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "data/processed/cascade_train_network.csv"
TEST_PATH = "data/processed/cascade_test_network.csv"

TARGET = "Target"

# Feature set frozen during Phase 9.4.
FEATURES = [
    "TimeTo5",
    "MinInterarrival",
    "EarlyAcceleration",
    "NeighbourhoodReach",
    "EarlyDensity",
    "MeanClustering",
    "CommunityDiversity",
    "MeanPageRank",
]

N_BOOTSTRAPS = 2000
RANDOM_STATE = 42


# ============================================================
# Load Data
# ============================================================

def load_data():
    print("Loading training and untouched test datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training shape: {train_df.shape}")
    print(f"Test shape:     {test_df.shape}")

    return train_df, test_df


# ============================================================
# Validate Data
# ============================================================

def validate_data(train_df, test_df):
    required_columns = FEATURES + [TARGET]

    for name, df in [
        ("Training", train_df),
        ("Test", test_df),
    ]:
        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{name} dataset is missing columns: "
                f"{missing_columns}"
            )

        if df[required_columns].isna().any().any():
            raise ValueError(
                f"{name} dataset contains missing values."
            )

    print("Dataset validation: PASSED")


# ============================================================
# Build Final Frozen Model
# ============================================================

def build_final_model():
    """
    Final model selected during Phase 9.4 using validation
    evidence only.

    No test-set information was used to select these settings.
    """

    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Calculate Metrics
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities,
):
    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        ),
        "PR_AUC": average_precision_score(
            y_true,
            probabilities
        ),
    }


# ============================================================
# Bootstrap 95% Confidence Intervals
# ============================================================

def bootstrap_confidence_intervals(
    y_true,
    predictions,
    probabilities,
    n_bootstraps=N_BOOTSTRAPS,
):
    print("\nCalculating bootstrap confidence intervals...")
    print(f"Bootstrap samples: {n_bootstraps}")

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    rng = np.random.default_rng(RANDOM_STATE)

    n_samples = len(y_true)

    bootstrap_results = []

    for _ in range(n_bootstraps):

        indices = rng.integers(
            0,
            n_samples,
            size=n_samples
        )

        y_boot = y_true[indices]

        # ROC-AUC requires both target classes.
        if len(np.unique(y_boot)) < 2:
            continue

        predictions_boot = predictions[indices]
        probabilities_boot = probabilities[indices]

        bootstrap_results.append(
            calculate_metrics(
                y_boot,
                predictions_boot,
                probabilities_boot,
            )
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )

    confidence_intervals = {}

    for metric in bootstrap_df.columns:

        lower = bootstrap_df[
            metric
        ].quantile(0.025)

        upper = bootstrap_df[
            metric
        ].quantile(0.975)

        confidence_intervals[metric] = (
            lower,
            upper
        )

    return confidence_intervals


# ============================================================
# Final Test Evaluation
# ============================================================

def evaluate_final_model(
    model,
    X_test,
    y_test,
):
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    confidence_intervals = (
        bootstrap_confidence_intervals(
            y_test,
            predictions,
            probabilities,
        )
    )

    print("\n" + "=" * 90)
    print(
        "FINAL BALANCED RANDOM FOREST — "
        "HELD-OUT TEST SET"
    )
    print("=" * 90)

    print(
        "\nTest metrics with 95% "
        "bootstrap confidence intervals:\n"
    )

    for metric, estimate in metrics.items():

        lower, upper = confidence_intervals[
            metric
        ]

        print(
            f"{metric:10s}: "
            f"{estimate:.4f} "
            f"[{lower:.4f}, {upper:.4f}]"
        )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return (
        metrics,
        confidence_intervals,
        predictions,
        probabilities,
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load and validate data
    # --------------------------------------------------------

    train_df, test_df = load_data()

    validate_data(
        train_df,
        test_df
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # --------------------------------------------------------
    # Test-set distribution
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("HELD-OUT TEST CLASS DISTRIBUTION")
    print("=" * 90)

    print(y_test.value_counts())

    print("\nProportions:")
    print(
        y_test.value_counts(
            normalize=True
        ).round(4)
    )

    # --------------------------------------------------------
    # Final frozen model
    # --------------------------------------------------------

    print("\nFinal model configuration:")
    print("Model:               Balanced Random Forest")
    print("Features:            8")
    print("n_estimators:        500")
    print("class_weight:        balanced")
    print("classification cut:  0.5")
    print("random_state:        42")

    model = build_final_model()

    print(
        "\nTraining final frozen model "
        "on training data..."
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Final held-out evaluation
    # --------------------------------------------------------

    evaluate_final_model(
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()