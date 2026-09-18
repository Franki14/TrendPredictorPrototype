from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
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
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "cascade_train_network.csv"
VALID_PATH = PROCESSED_DIR / "cascade_validation_network.csv"
TEST_PATH = PROCESSED_DIR / "cascade_test_network.csv"


# ============================================================
# Final Feature Set
# ============================================================

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

TARGET = "Target"


# ============================================================
# Data Loading
# ============================================================

def load_data():
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {valid_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    return train_df, valid_df, test_df


# ============================================================
# Validation
# ============================================================

def validate_data(train_df, valid_df, test_df):
    required_columns = FEATURES + [TARGET]

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", valid_df),
        ("TEST", test_df),
    ]:
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"{name} dataset is missing required columns: {missing}"
            )

        missing_values = df[required_columns].isna().sum().sum()

        if missing_values != 0:
            raise ValueError(
                f"{name} contains {missing_values} missing values."
            )

    print("Dataset validation: PASSED")


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(model, X, y, dataset_name):
    predictions = model.predict(X)

    # DummyClassifier supports predict_proba.
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    # These metrics use probability scores rather than hard predictions.
    roc_auc = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)

    cm = confusion_matrix(y, predictions)

    print("\n" + "=" * 80)
    print(dataset_name)
    print("=" * 80)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


# ============================================================
# Main
# ============================================================

def main():

    train_df, valid_df, test_df = load_data()

    validate_data(train_df, valid_df, test_df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\n" + "=" * 80)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 80)

    print(y_train.value_counts())
    print("\nProportions:")
    print(y_train.value_counts(normalize=True).round(4))


    # ========================================================
    # Baseline 1 - Most Frequent Class
    # ========================================================

    print("\n\n" + "#" * 80)
    print("BASELINE 1 — MOST FREQUENT CLASS")
    print("#" * 80)

    most_frequent = DummyClassifier(
        strategy="most_frequent"
    )

    most_frequent.fit(X_train, y_train)

    results = []

    results.append(
        evaluate_model(
            most_frequent,
            X_valid,
            y_valid,
            "Most Frequent — Validation",
        )
    )

    results.append(
        evaluate_model(
            most_frequent,
            X_test,
            y_test,
            "Most Frequent — Test",
        )
    )


    # ========================================================
    # Baseline 2 - Stratified
    # ========================================================

    print("\n\n" + "#" * 80)
    print("BASELINE 2 — STRATIFIED RANDOM")
    print("#" * 80)

    stratified = DummyClassifier(
        strategy="stratified",
        random_state=42,
    )

    stratified.fit(X_train, y_train)

    results.append(
        evaluate_model(
            stratified,
            X_valid,
            y_valid,
            "Stratified — Validation",
        )
    )

    results.append(
        evaluate_model(
            stratified,
            X_test,
            y_test,
            "Stratified — Test",
        )
    )


    # ========================================================
    # Summary
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 80)
    print("BASELINE MODEL SUMMARY")
    print("=" * 80)

    print(
        results_df.round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()