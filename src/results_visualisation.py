"""
Final Validation Performance Visualisations

Generates report-ready visualisations for the established validation
experiments.

Figures:
1. ROC curve — Logistic Regression vs Random Forest
2. Precision-Recall curve — Logistic Regression vs Random Forest
3. Validation confusion matrices
4. Main validation model comparison

IMPORTANT:
- Models use the already-established configurations.
- No hyperparameter selection is performed.
- The held-out test set is not used.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "data/processed/cascade_train_network.csv"
VALID_PATH = "data/processed/cascade_validation_network.csv"

FIGURES_DIR = "results/figures"

TARGET = "Target"

RANDOM_STATE = 42

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

    for name, dataframe in [
        ("training", train_df),
        ("validation", valid_df),
    ]:

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in {name} dataset: "
                f"{missing_columns}"
            )

        if dataframe[required_columns].isnull().any().any():
            raise ValueError(
                f"Missing values detected in {name} dataset."
            )

    print("Dataset validation: PASSED")


# ============================================================
# Build Models
# ============================================================

def build_logistic_regression():

    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_random_forest():

    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Train Models
# ============================================================

def train_models(
    X_train,
    y_train,
):

    print("\nTraining established models...")

    lr_model = build_logistic_regression()
    rf_model = build_random_forest()

    print("Training Balanced Logistic Regression...")
    lr_model.fit(
        X_train,
        y_train,
    )

    print("Training Balanced Random Forest...")
    rf_model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    return lr_model, rf_model


# ============================================================
# Generate Validation Predictions
# ============================================================

def generate_predictions(
    lr_model,
    rf_model,
    X_valid,
):

    lr_predictions = lr_model.predict(X_valid)
    lr_probabilities = lr_model.predict_proba(
        X_valid
    )[:, 1]

    rf_predictions = rf_model.predict(X_valid)
    rf_probabilities = rf_model.predict_proba(
        X_valid
    )[:, 1]

    return {
        "Balanced Logistic Regression": {
            "predictions": lr_predictions,
            "probabilities": lr_probabilities,
        },
        "Balanced Random Forest": {
            "predictions": rf_predictions,
            "probabilities": rf_probabilities,
        },
    }


# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(
    y_valid,
    model_outputs,
):

    print("\nGenerating ROC curve...")

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for model_name, output in model_outputs.items():

        probabilities = output["probabilities"]

        fpr, tpr, _ = roc_curve(
            y_valid,
            probabilities,
        )

        roc_auc = roc_auc_score(
            y_valid,
            probabilities,
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{model_name} "
                f"(AUC = {roc_auc:.3f})"
            ),
        )

    # Random classifier reference line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier (AUC = 0.500)",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Validation ROC Curves"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "validation_roc_curve.png",
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

def plot_precision_recall_curve(
    y_valid,
    model_outputs,
):

    print(
        "Generating Precision-Recall curve..."
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for model_name, output in model_outputs.items():

        probabilities = output["probabilities"]

        precision, recall, _ = (
            precision_recall_curve(
                y_valid,
                probabilities,
            )
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities,
        )

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=(
                f"{model_name} "
                f"(AP = {pr_auc:.3f})"
            ),
        )

    # Positive-class prevalence is the no-skill PR baseline.
    prevalence = np.mean(y_valid)

    ax.axhline(
        prevalence,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"No-skill baseline "
            f"({prevalence:.3f})"
        ),
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Validation Precision-Recall Curves"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.legend(
        loc="upper right"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "validation_pr_curve.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Individual Confusion Matrix Plot
# ============================================================

def draw_confusion_matrix(
    ax,
    matrix,
    title,
):

    image = ax.imshow(
        matrix,
        interpolation="nearest",
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        ["Non-emergent", "Emergent"]
    )

    ax.set_yticklabels(
        ["Non-emergent", "Emergent"]
    )

    threshold = (
        matrix.max() / 2
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = matrix[
                row,
                column
            ]

            text_colour = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                column,
                row,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color=text_colour,
                fontsize=12,
            )

    return image


# ============================================================
# Confusion Matrices
# ============================================================

def plot_confusion_matrices(
    y_valid,
    model_outputs,
):

    print(
        "Generating confusion matrices..."
    )

    # This is intentionally one combined figure because the
    # two matrices are directly compared side-by-side.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.8),
    )

    model_names = [
        "Balanced Logistic Regression",
        "Balanced Random Forest",
    ]

    images = []

    for ax, model_name in zip(
        axes,
        model_names,
    ):

        predictions = (
            model_outputs[
                model_name
            ]["predictions"]
        )

        matrix = confusion_matrix(
            y_valid,
            predictions,
        )

        image = draw_confusion_matrix(
            ax,
            matrix,
            model_name,
        )

        images.append(image)

    fig.suptitle(
        "Validation Confusion Matrices",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "validation_confusion_matrices.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Main Model Comparison
# ============================================================

def plot_model_comparison():

    print(
        "Generating main model comparison..."
    )

    comparison_path = (
        "results/tables/"
        "validation_model_comparison.csv"
    )

    if not os.path.exists(
        comparison_path
    ):
        raise FileNotFoundError(
            "Validation model comparison table "
            "not found. Run "
            "'python src/results_summary.py' first."
        )

    results_df = pd.read_csv(
        comparison_path
    )

    metrics = [
        "F1",
        "ROC_AUC",
        "PR_AUC",
    ]

    plot_df = (
        results_df[
            ["Model"] + metrics
        ]
        .set_index("Model")
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.75,
    )

    ax.set_title(
        "Validation Model Performance Comparison"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.legend(
        title="Metric"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # Add values above bars.
    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.3f",
            padding=3,
            fontsize=8,
        )

    fig = ax.get_figure()

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "validation_model_comparison.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# ============================================================
# Print Verification Metrics
# ============================================================

def verify_metrics(
    y_valid,
    model_outputs,
):

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION CURVE METRIC VERIFICATION"
    )

    print(
        "=" * 80
    )

    for model_name, output in model_outputs.items():

        probabilities = output[
            "probabilities"
        ]

        roc_auc = roc_auc_score(
            y_valid,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities,
        )

        matrix = confusion_matrix(
            y_valid,
            output["predictions"],
        )

        print(
            f"\n{model_name}"
        )

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"PR-AUC:  {pr_auc:.4f}"
        )

        print(
            "Confusion matrix:"
        )

        print(matrix)

# ============================================================
# Feature-Group Ablation Visualisation
# ============================================================

def plot_ablation_results():

    print(
        "Generating feature-group ablation visualisation..."
    )

    ablation_path = (
        "results/tables/"
        "feature_group_ablation.csv"
    )

    if not os.path.exists(ablation_path):
        raise FileNotFoundError(
            "Feature-group ablation table not found. "
            "Run 'python src/results_summary.py' first."
        )

    ablation_df = pd.read_csv(
        ablation_path
    )

    metrics = [
        "F1",
        "ROC_AUC",
        "PR_AUC",
    ]

    plot_df = (
        ablation_df[
            ["FeatureGroup"] + metrics
        ]
        .set_index("FeatureGroup")
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.72,
    )

    ax.set_title(
        "Feature-Group Ablation Study"
    )

    ax.set_xlabel(
        "Feature Group"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    ax.legend(
        title="Metric"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.3f",
            padding=3,
            fontsize=9,
        )

    fig = ax.get_figure()

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "feature_group_ablation.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )

# ============================================================
# Paired Bootstrap Statistical Comparison
# ============================================================

def plot_paired_bootstrap_comparison():

    print(
        "Generating paired-bootstrap comparison visualisation..."
    )

    bootstrap_path = (
        "results/tables/"
        "paired_bootstrap_comparison.csv"
    )

    if not os.path.exists(bootstrap_path):
        raise FileNotFoundError(
            "Paired-bootstrap comparison table not found. "
            "Run 'python src/results_summary.py' first."
        )

    bootstrap_df = pd.read_csv(
        bootstrap_path
    )

    # --------------------------------------------------------
    # Prepare values
    # --------------------------------------------------------

    metrics = bootstrap_df["Metric"].copy()

    # Improve labels for the final figure.
    metrics = metrics.replace({
        "ROC_AUC": "ROC-AUC",
        "PR_AUC": "PR-AUC",
    })

    differences = bootstrap_df[
        "RF_minus_LR"
    ].to_numpy()

    lower_bounds = bootstrap_df[
        "CI_Lower"
    ].to_numpy()

    upper_bounds = bootstrap_df[
        "CI_Upper"
    ].to_numpy()

    # Convert CI bounds into distances from estimate.
    lower_errors = (
        differences - lower_bounds
    )

    upper_errors = (
        upper_bounds - differences
    )

    error_values = np.vstack([
        lower_errors,
        upper_errors,
    ])

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    positions = np.arange(
        len(metrics)
    )

    ax.errorbar(
        differences,
        positions,
        xerr=error_values,
        fmt="o",
        markersize=7,
        capsize=5,
        linewidth=1.8,
    )

    # Zero = no difference between models.
    ax.axvline(
        x=0,
        linestyle="--",
        linewidth=1.5,
        label="No difference",
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        metrics
    )

    ax.set_xlabel(
        "Performance Difference "
        "(Random Forest − Logistic Regression)"
    )

    ax.set_ylabel(
        "Metric"
    )

    ax.set_title(
    "Paired Bootstrap Model Comparison (95% CI)"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    # Put first metric at top.
    ax.invert_yaxis()

    # --------------------------------------------------------
    # Add numerical values
    # --------------------------------------------------------

    for position, estimate, lower, upper in zip(
        positions,
        differences,
        lower_bounds,
        upper_bounds,
    ):

        label = (
            f"{estimate:+.3f} "
            f"[{lower:+.3f}, {upper:+.3f}]"
        )

        if estimate >= 0:

            text_x = upper + 0.01
            alignment = "left"

        else:

            text_x = lower - 0.01
            alignment = "right"

        ax.text(
            text_x,
            position,
            label,
            va="center",
            ha=alignment,
            fontsize=9,
        )

    # Give labels enough horizontal room.
    minimum_x = min(
        lower_bounds.min(),
        0
    )

    maximum_x = max(
        upper_bounds.max(),
        0
    )

    margin = (
        maximum_x - minimum_x
    ) * 0.45

    ax.set_xlim(
        minimum_x - margin,
        maximum_x + margin,
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "paired_bootstrap_comparison.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )

# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n" + "#" * 80
    )

    print(
        "FINAL PERFORMANCE VISUALISATIONS"
    )

    print(
        "#" * 80
    )

    os.makedirs(
        FIGURES_DIR,
        exist_ok=True,
    )

    # ========================================================
    # Data
    # ========================================================

    train_df, valid_df = load_data()

    validate_data(
        train_df,
        valid_df,
    )

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_valid = valid_df[
        FEATURES
    ]

    y_valid = valid_df[
        TARGET
    ]

    # ========================================================
    # Established Models
    # ========================================================

    lr_model, rf_model = train_models(
        X_train,
        y_train,
    )

    # ========================================================
    # Predictions
    # ========================================================

    model_outputs = generate_predictions(
        lr_model,
        rf_model,
        X_valid,
    )

    # ========================================================
    # Verify Results
    # ========================================================

    verify_metrics(
        y_valid,
        model_outputs,
    )

    # ========================================================
    # Figures
    # ========================================================

    plot_roc_curve(
        y_valid,
        model_outputs,
    )

    plot_precision_recall_curve(
        y_valid,
        model_outputs,
    )

    plot_confusion_matrices(
        y_valid,
        model_outputs,
    )

    plot_model_comparison()

    plot_ablation_results()

    plot_paired_bootstrap_comparison()

    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "VISUALISATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nFigures saved to: {FIGURES_DIR}/"
    )


if __name__ == "__main__":
    main()