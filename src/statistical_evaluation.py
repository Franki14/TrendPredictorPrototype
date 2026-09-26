
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "data/processed/cascade_train_network.csv"
VALID_PATH = "data/processed/cascade_validation_network.csv"

TARGET = "Target"

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

    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    valid_df = pd.read_csv(VALID_PATH)

    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {valid_df.shape}")

    required = FEATURES + [TARGET]

    for name, df in [
        ("Training", train_df),
        ("Validation", valid_df),
    ]:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if df[required].isna().any().any():
            raise ValueError(
                f"{name} dataset contains missing values."
            )

    print("Dataset validation: PASSED")

    return train_df, valid_df


# ============================================================
# Train Models
# ============================================================

def train_models(
    X_train,
    y_train
):

    print("\nTraining models...")

    logistic_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
        ),
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    print("Model training complete.")

    return {
        "Balanced Logistic Regression": logistic_model,
        "Balanced Random Forest": random_forest_model,
    }


# ============================================================
# Metric Calculation
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities
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
# Bootstrap Confidence Intervals
# ============================================================

def bootstrap_confidence_intervals(
    y_true,
    predictions,
    probabilities,
    n_bootstraps=N_BOOTSTRAPS,
    random_state=RANDOM_STATE,
):

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    rng = np.random.default_rng(
        random_state
    )

    n_samples = len(y_true)

    bootstrap_results = []

    for _ in range(n_bootstraps):

        indices = rng.integers(
            0,
            n_samples,
            size=n_samples
        )

        y_boot = y_true[indices]

        # ROC-AUC requires both classes.
        if len(np.unique(y_boot)) < 2:
            continue

        pred_boot = predictions[indices]
        prob_boot = probabilities[indices]

        metrics = calculate_metrics(
            y_boot,
            pred_boot,
            prob_boot
        )

        bootstrap_results.append(
            metrics
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
# Evaluate Models
# ============================================================

def evaluate_models(
    models,
    X_valid,
    y_valid
):

    summary_rows = []

    for model_name, model in models.items():

        print("\n" + "=" * 90)
        print(model_name.upper())
        print("=" * 90)

        predictions = model.predict(
            X_valid
        )

        probabilities = model.predict_proba(
            X_valid
        )[:, 1]

        point_metrics = calculate_metrics(
            y_valid,
            predictions,
            probabilities
        )

        print("\nCalculating bootstrap confidence intervals...")

        confidence_intervals = (
            bootstrap_confidence_intervals(
                y_valid,
                predictions,
                probabilities,
            )
        )

        print(
            f"Bootstrap samples: {N_BOOTSTRAPS}"
        )

        print("\nValidation metrics with 95% bootstrap CIs:\n")

        for metric, value in point_metrics.items():

            lower, upper = (
                confidence_intervals[metric]
            )

            print(
                f"{metric:10s}: "
                f"{value:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )

            summary_rows.append({
                "Model": model_name,
                "Metric": metric,
                "Estimate": value,
                "CI_Lower": lower,
                "CI_Upper": upper,
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    return summary_df

# ============================================================
# Paired Bootstrap Model Comparison
# ============================================================

def paired_bootstrap_comparison(
    y_true,
    lr_predictions,
    lr_probabilities,
    rf_predictions,
    rf_probabilities,
    n_bootstraps=N_BOOTSTRAPS,
    random_state=RANDOM_STATE,
):

    y_true = np.asarray(y_true)

    lr_predictions = np.asarray(
        lr_predictions
    )

    lr_probabilities = np.asarray(
        lr_probabilities
    )

    rf_predictions = np.asarray(
        rf_predictions
    )

    rf_probabilities = np.asarray(
        rf_probabilities
    )

    rng = np.random.default_rng(
        random_state
    )

    n_samples = len(y_true)

    bootstrap_differences = []

    for _ in range(n_bootstraps):

        indices = rng.integers(
            0,
            n_samples,
            size=n_samples
        )

        y_boot = y_true[indices]

        # AUC metrics require both classes.
        if len(np.unique(y_boot)) < 2:
            continue

        lr_metrics = calculate_metrics(
            y_boot,
            lr_predictions[indices],
            lr_probabilities[indices],
        )

        rf_metrics = calculate_metrics(
            y_boot,
            rf_predictions[indices],
            rf_probabilities[indices],
        )

        differences = {
            metric:
                rf_metrics[metric]
                - lr_metrics[metric]
            for metric in lr_metrics
        }

        bootstrap_differences.append(
            differences
        )

    difference_df = pd.DataFrame(
        bootstrap_differences
    )

    # --------------------------------------------------------
    # Observed differences on original validation set
    # --------------------------------------------------------

    lr_observed = calculate_metrics(
        y_true,
        lr_predictions,
        lr_probabilities,
    )

    rf_observed = calculate_metrics(
        y_true,
        rf_predictions,
        rf_probabilities,
    )

    results = []

    for metric in difference_df.columns:

        observed_difference = (
            rf_observed[metric]
            - lr_observed[metric]
        )

        lower = difference_df[
            metric
        ].quantile(0.025)

        upper = difference_df[
            metric
        ].quantile(0.975)

        # Two-sided bootstrap sign-based p-value.
        proportion_le_zero = (
            difference_df[metric] <= 0
        ).mean()

        proportion_ge_zero = (
            difference_df[metric] >= 0
        ).mean()

        p_value = min(
            1.0,
            2 * min(
                proportion_le_zero,
                proportion_ge_zero,
            )
        )

        results.append({
            "Metric": metric,
            "RF_minus_LR": observed_difference,
            "CI_Lower": lower,
            "CI_Upper": upper,
            "P_Value": p_value,
            "CI_Excludes_Zero": (
                lower > 0 or upper < 0
            ),
        })

    return pd.DataFrame(results)

# ============================================================
# McNemar's Test
# ============================================================

def mcnemar_model_comparison(
    y_true,
    lr_predictions,
    rf_predictions,
):

    y_true = np.asarray(y_true)
    lr_predictions = np.asarray(lr_predictions)
    rf_predictions = np.asarray(rf_predictions)

    lr_correct = (
        lr_predictions == y_true
    )

    rf_correct = (
        rf_predictions == y_true
    )

    # Paired correctness table:
    #
    #                         RF
    #                   Correct  Wrong
    # LR Correct           a       b
    # LR Wrong             c       d

    both_correct = np.sum(
        lr_correct & rf_correct
    )

    lr_correct_rf_wrong = np.sum(
        lr_correct & ~rf_correct
    )

    lr_wrong_rf_correct = np.sum(
        ~lr_correct & rf_correct
    )

    both_wrong = np.sum(
        ~lr_correct & ~rf_correct
    )

    table = np.array([
        [
            both_correct,
            lr_correct_rf_wrong
        ],
        [
            lr_wrong_rf_correct,
            both_wrong
        ],
    ])

    print("\n" + "=" * 80)
    print("MCNEMAR'S TEST — LR VS RF")
    print("=" * 80)

    print("\nPaired correctness table:")
    print("\n                     RF Correct    RF Wrong")
    print(
        f"LR Correct          "
        f"{both_correct:10d}    "
        f"{lr_correct_rf_wrong:8d}"
    )
    print(
        f"LR Wrong            "
        f"{lr_wrong_rf_correct:10d}    "
        f"{both_wrong:8d}"
    )

    print("\nDiscordant pairs:")
    print(
        "LR correct / RF wrong:",
        lr_correct_rf_wrong
    )
    print(
        "LR wrong / RF correct:",
        lr_wrong_rf_correct
    )

    # Exact McNemar test is appropriate and avoids relying
    # on the large-sample chi-square approximation.
    result = mcnemar(
        table,
        exact=True,
    )

    print("\nExact McNemar test:")
    print(
        f"Statistic: {result.statistic:.4f}"
    )
    print(
        f"P-value:   {result.pvalue:.6f}"
    )

    if result.pvalue < 0.05:
        print(
            "\nResult: Significant difference "
            "in paired classification errors "
            "(p < 0.05)."
        )
    else:
        print(
            "\nResult: No statistically significant "
            "difference in paired classification "
            "errors (p >= 0.05)."
        )

    return {
        "BothCorrect": both_correct,
        "LRCorrect_RFIncorrect": lr_correct_rf_wrong,
        "LRIncorrect_RFCorrect": lr_wrong_rf_correct,
        "BothIncorrect": both_wrong,
        "Statistic": result.statistic,
        "PValue": result.pvalue,
    }

# ============================================================
# Main
# ============================================================

def main():

    train_df, valid_df = load_data()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    models = train_models(
        X_train,
        y_train
    )

    summary_df = evaluate_models(
        models,
        X_valid,
        y_valid
    )

    print("\n" + "=" * 100)
    print("BOOTSTRAP CONFIDENCE INTERVAL SUMMARY")
    print("=" * 100)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

        # ========================================================
    # Paired Bootstrap Comparison
    # ========================================================

    print("\n" + "#" * 100)
    print("PAIRED BOOTSTRAP COMPARISON")
    print("Balanced Random Forest minus Balanced Logistic Regression")
    print("#" * 100)

    lr_model = models[
        "Balanced Logistic Regression"
    ]

    rf_model = models[
        "Balanced Random Forest"
    ]

    lr_predictions = lr_model.predict(
        X_valid
    )

    lr_probabilities = lr_model.predict_proba(
        X_valid
    )[:, 1]

    rf_predictions = rf_model.predict(
        X_valid
    )

    rf_probabilities = rf_model.predict_proba(
        X_valid
    )[:, 1]

    comparison_df = paired_bootstrap_comparison(
        y_valid,
        lr_predictions,
        lr_probabilities,
        rf_predictions,
        rf_probabilities,
    )

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ========================================================
    # McNemar's Test
    # ========================================================

    mcnemar_results = mcnemar_model_comparison(
        y_valid,
        lr_predictions,
        rf_predictions,
    )


if __name__ == "__main__":
    main()