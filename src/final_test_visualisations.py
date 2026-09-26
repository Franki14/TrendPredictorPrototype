# ============================================================
# Final Held-Out Test Visualisations
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from final_test_evaluation import (
    TRAIN_PATH,
    TEST_PATH,
    FEATURES,
    TARGET,
    RANDOM_STATE,
    build_final_model,
    calculate_metrics,
    bootstrap_confidence_intervals,
)


# ============================================================
# Configuration
# ============================================================

FIGURE_DIR = "results/figures"

os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)


# ============================================================
# Load Data
# ============================================================

def load_data():

    print("Loading training and held-out test datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training shape: {train_df.shape}")
    print(f"Test shape:     {test_df.shape}")

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

    return train_df, test_df


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(
    y_test,
    predictions,
):

    print("Generating final test confusion matrix...")

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm,
        cmap="Blues",
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    class_labels = [
        "Non-emergent",
        "Emergent",
    ]

    ax.set_xticks(
        np.arange(2)
    )

    ax.set_yticks(
        np.arange(2)
    )

    ax.set_xticklabels(
        class_labels
    )

    ax.set_yticklabels(
        class_labels
    )

    threshold = (
        cm.max() / 2
    )

    for row in range(cm.shape[0]):

        for column in range(cm.shape[1]):

            value = cm[row, column]

            text_colour = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=16,
                color=text_colour,
            )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "Final Balanced Random Forest\n"
        "Held-Out Test Confusion Matrix"
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "final_test_confusion_matrix.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(
    y_test,
    probabilities,
):

    print("Generating final test ROC curve...")

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_test,
            probabilities,
        )
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=(
            "Balanced Random Forest "
            f"(AUC = {roc_auc:.3f})"
        ),
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier (AUC = 0.500)",
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Final Held-Out Test ROC Curve"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "final_test_roc_curve.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Precision-Recall Curve
# ============================================================

def plot_pr_curve(
    y_test,
    probabilities,
):

    print(
        "Generating final test "
        "Precision-Recall curve..."
    )

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            probabilities,
        )
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    positive_prevalence = (
        np.mean(y_test)
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=(
            "Balanced Random Forest "
            f"(AP = {pr_auc:.3f})"
        ),
    )

    ax.axhline(
        positive_prevalence,
        linestyle="--",
        linewidth=1.5,
        label=(
            "No-skill baseline "
            f"({positive_prevalence:.3f})"
        ),
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Final Held-Out Test Precision-Recall Curve"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "final_test_pr_curve.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Test Metrics with Bootstrap Confidence Intervals
# ============================================================

def plot_metrics_with_confidence_intervals(
    metrics,
    confidence_intervals,
):

    print(
        "Generating final test metric "
        "confidence interval plot..."
    )

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
    ]

    display_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ]

    estimates = np.array([
        metrics[metric]
        for metric in metric_names
    ])

    lower_bounds = np.array([
        confidence_intervals[metric][0]
        for metric in metric_names
    ])

    upper_bounds = np.array([
        confidence_intervals[metric][1]
        for metric in metric_names
    ])

    lower_errors = (
        estimates - lower_bounds
    )

    upper_errors = (
        upper_bounds - estimates
    )

    error_values = np.vstack([
        lower_errors,
        upper_errors,
    ])

    y_positions = np.arange(
        len(metric_names)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.errorbar(
        estimates,
        y_positions,
        xerr=error_values,
        fmt="o",
        markersize=7,
        capsize=5,
        linewidth=1.5,
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        display_names
    )

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        1,
    )

    ax.set_xlabel(
        "Score"
    )

    ax.set_ylabel(
        "Metric"
    )

    ax.set_title(
        "Final Held-Out Test Performance\n"
        "with 95% Bootstrap Confidence Intervals"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    for index, estimate in enumerate(estimates):

        lower = lower_bounds[index]
        upper = upper_bounds[index]

        annotation = (
            f"{estimate:.3f} "
            f"[{lower:.3f}, {upper:.3f}]"
        )

        annotation_x = min(
            upper + 0.025,
            0.83,
        )

        ax.text(
            annotation_x,
            index,
            annotation,
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "final_test_metrics_ci.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "#" * 80)
    print("FINAL HELD-OUT TEST VISUALISATIONS")
    print("#" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df, test_df = load_data()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # --------------------------------------------------------
    # Build the already-established frozen model
    # --------------------------------------------------------

    print(
        "\nTraining established frozen "
        "Balanced Random Forest..."
    )

    model = build_final_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Generate frozen test predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Verify metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL TEST METRIC VERIFICATION"
    )

    print(
        "=" * 80
    )

    for metric, value in metrics.items():

        print(
            f"{metric:10s}: {value:.4f}"
        )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "\nConfusion matrix:"
    )

    print(cm)

    # --------------------------------------------------------
    # Bootstrap confidence intervals
    # --------------------------------------------------------

    confidence_intervals = (
        bootstrap_confidence_intervals(
            y_test,
            predictions,
            probabilities,
        )
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    plot_confusion_matrix(
        y_test,
        predictions,
    )

    plot_roc_curve(
        y_test,
        probabilities,
    )

    plot_pr_curve(
        y_test,
        probabilities,
    )

    plot_metrics_with_confidence_intervals(
        metrics,
        confidence_intervals,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL TEST VISUALISATION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()