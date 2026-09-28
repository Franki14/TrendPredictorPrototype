from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "cascade_train_network.csv"
VALID_PATH = PROCESSED_DIR / "cascade_validation_network.csv"


# ============================================================
# Feature Set
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
# Load Data
# ============================================================

def load_data():

    print("Loading training and validation datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)

    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {valid_df.shape}")

    return train_df, valid_df


# ============================================================
# Validate Data
# ============================================================

def validate_data(train_df, valid_df):

    required_columns = FEATURES + [TARGET]

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", valid_df),
    ]:

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{name} missing columns: {missing_columns}"
            )

        missing_values = (
            df[required_columns]
            .isna()
            .sum()
            .sum()
        )

        if missing_values != 0:
            raise ValueError(
                f"{name} contains {missing_values} missing values."
            )

    print("Dataset validation: PASSED")


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y, probabilities
        ),
        "PR_AUC": average_precision_score(
            y, probabilities
        ),
    }

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION — VALIDATION RESULTS")
    print("=" * 80)

    for metric, value in results.items():
        print(f"{metric:10}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return results


# ============================================================
# Coefficient Analysis
# ============================================================

def analyse_coefficients(model):

    logistic_model = model.named_steps["classifier"]

    coefficients = logistic_model.coef_[0]

    coefficient_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": coefficients,
        "OddsRatio": np.exp(coefficients),
    })

    coefficient_df["AbsCoefficient"] = (
        coefficient_df["Coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        "AbsCoefficient",
        ascending=False,
    )

    print("\n" + "=" * 80)
    print("STANDARDISED LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 80)

    print(
        coefficient_df[
            [
                "Feature",
                "Coefficient",
                "OddsRatio",
            ]
        ].round(4).to_string(index=False)
    )

    return coefficient_df

# ============================================================
# Class Weight Experiment
# ============================================================

def compare_class_weights(
    X_train,
    y_train,
    X_val,
    y_val
):
    """
    Compare unweighted and class-balanced Logistic Regression models.

    Everything except class_weight is kept identical so that the
    effect of class balancing can be evaluated fairly.
    """

    experiments = {
        "Unweighted": None,
        "Balanced": "balanced"
    }

    results = []

    print("\n" + "#" * 80)
    print("CLASS-WEIGHT EXPERIMENT")
    print("#" * 80)

    for name, class_weight in experiments.items():

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight=class_weight,
                max_iter=2000,
                random_state=42
            ))
        ])

        print(f"\nTraining {name} Logistic Regression...")

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)
        probabilities = pipeline.predict_proba(X_val)[:, 1]

        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )
        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )
        roc_auc = roc_auc_score(
            y_val,
            probabilities
        )
        pr_auc = average_precision_score(
            y_val,
            probabilities
        )

        cm = confusion_matrix(
            y_val,
            predictions
        )

        print("\n" + "=" * 80)
        print(f"{name.upper()} LOGISTIC REGRESSION")
        print("=" * 80)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1        : {f1:.4f}")
        print(f"ROC_AUC   : {roc_auc:.4f}")
        print(f"PR_AUC    : {pr_auc:.4f}")

        print("\nConfusion matrix:")
        print(cm)

        print("\nClassification report:")
        print(
            classification_report(
                y_val,
                predictions,
                digits=4,
                zero_division=0
            )
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1]
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("CLASS-WEIGHT COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return results_df


# ============================================================
# Main
# ============================================================

def main():

    train_df, valid_df = load_data()

    validate_data(train_df, valid_df)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    # ========================================================
    # Class-Weight Experiment
    # ========================================================

    comparison_df = compare_class_weights(
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    # ========================================================
    # Pipeline — Balanced Logistic Regression
    # ========================================================

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            )
        ),
    ])

    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    print("Training complete.")

    # ========================================================
    # Validation Evaluation
    # ========================================================

    results = evaluate_model(
        model,
        X_valid,
        y_valid,
    )

    # ========================================================
    # Interpretability
    # ========================================================

    coefficient_df = analyse_coefficients(model)


if __name__ == "__main__":
    main()