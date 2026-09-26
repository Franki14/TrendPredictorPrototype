import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import (
    permutation_importance,
    partial_dependence,
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

RANDOM_STATE = 42

FIGURE_DIR = "results/figures"

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
        ("Training", train_df),
        ("Validation", valid_df),
    ]:

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if df[required_columns].isna().any().any():
            raise ValueError(
                f"{name} dataset contains missing values."
            )

    print("Dataset validation: PASSED")


# ============================================================
# Build Frozen Random Forest
# ============================================================

def build_model():

    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# Built-in Random Forest Importance
# ============================================================

def calculate_impurity_importance(model):

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "ImpurityImportance":
            model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "ImpurityImportance",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("RANDOM FOREST IMPURITY FEATURE IMPORTANCE")
    print("=" * 80)

    print(
        importance_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return importance_df


# ============================================================
# Permutation Importance
# ============================================================

def calculate_permutation_importance(
    model,
    X_valid,
    y_valid,
):

    print("\nCalculating permutation importance...")

    result = permutation_importance(
        model,
        X_valid,
        y_valid,
        scoring="roc_auc",
        n_repeats=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    permutation_df = pd.DataFrame({
        "Feature": FEATURES,
        "PermutationMean":
            result.importances_mean,
        "PermutationStd":
            result.importances_std,
    })

    permutation_df = permutation_df.sort_values(
        "PermutationMean",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("PERMUTATION IMPORTANCE — VALIDATION ROC-AUC")
    print("=" * 80)

    print(
        permutation_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return permutation_df


# ============================================================
# Compare Importance Methods
# ============================================================

def compare_importance(
    impurity_df,
    permutation_df,
):

    comparison_df = pd.merge(
        impurity_df,
        permutation_df,
        on="Feature",
    )

    comparison_df[
        "ImpurityRank"
    ] = comparison_df[
        "ImpurityImportance"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    comparison_df[
        "PermutationRank"
    ] = comparison_df[
        "PermutationMean"
    ].rank(
        ascending=False,
        method="min",
    ).astype(int)

    comparison_df = comparison_df.sort_values(
        "PermutationRank"
    )

    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE COMPARISON")
    print("=" * 100)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    return comparison_df

# ============================================================
# Partial Dependence Analysis
# ============================================================

PDP_FEATURES = [
    "NeighbourhoodReach",
    "EarlyDensity",
    "EarlyAcceleration",
    "MinInterarrival",
]


def calculate_partial_dependence(
    model,
    X_train,
):

    print("\n" + "=" * 100)
    print("PARTIAL DEPENDENCE ANALYSIS")
    print("=" * 100)

    results = {}

    for feature in PDP_FEATURES:

        print(f"\nFeature: {feature}")

        pd_result = partial_dependence(
            model,
            X_train,
            features=[feature],
            kind="average",
            grid_resolution=20,
            percentiles=(0.05, 0.95),
            method="brute",
        )

        values = pd_result["grid_values"][0]

        average = pd_result[
            "average"
        ][0]

        feature_df = pd.DataFrame({
            "FeatureValue": values,
            "PartialDependence": average,
        })

        results[feature] = feature_df

        print(
            feature_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    return results

# ============================================================
# Two-Way Partial Dependence
# ============================================================

def calculate_interaction_partial_dependence(
    model,
    X_train,
):

    print("\n" + "=" * 100)
    print("TWO-WAY PARTIAL DEPENDENCE")
    print("NeighbourhoodReach × EarlyDensity")
    print("=" * 100)

    # Get column positions
    reach_idx = X_train.columns.get_loc(
        "NeighbourhoodReach"
    )

    density_idx = X_train.columns.get_loc(
        "EarlyDensity"
    )

    print(
        f"\nFeature indices: "
        f"NeighbourhoodReach={reach_idx}, "
        f"EarlyDensity={density_idx}"
    )

    # Calculate two-way PDP
    pd_result = partial_dependence(
        model,
        X_train,
        features=[(reach_idx, density_idx)],
        kind="average",
        grid_resolution=10,
        percentiles=(0.05, 0.95),
        method="brute",
    )

    reach_values = pd_result["grid_values"][0]
    density_values = pd_result["grid_values"][1]

    average = pd_result["average"][0]

    print("\nNeighbourhoodReach grid:")
    print(
        np.round(
            reach_values,
            3,
        )
    )

    print("\nEarlyDensity grid:")
    print(
        np.round(
            density_values,
            3,
        )
    )

    interaction_df = pd.DataFrame(
        average,
        index=np.round(reach_values, 3),
        columns=np.round(density_values, 3),
    )

    interaction_df.index.name = "NeighbourhoodReach"
    interaction_df.columns.name = "EarlyDensity"

    print("\nPartial dependence matrix:")
    print(
        interaction_df.round(4)
    )

    return {
        "NeighbourhoodReach": reach_values,
        "EarlyDensity": density_values,
        "PartialDependence": average,
    }

# ============================================================
# Plot Permutation Importance
# ============================================================

def plot_permutation_importance(
    permutation_df,
):

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True,
    )

    plot_df = permutation_df.sort_values(
        "PermutationMean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["Feature"],
        plot_df["PermutationMean"],
        xerr=plot_df["PermutationStd"],
        capsize=4,
    )

    ax.set_xlabel(
        "Mean Decrease in Validation ROC-AUC"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Random Forest Permutation Feature Importance"
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "permutation_importance.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved permutation importance figure: "
        f"{output_path}"
    )

    # ============================================================
# Plot One-Way Partial Dependence
# ============================================================

def plot_partial_dependence(
    pdp_results,
):

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True,
    )

    features_to_plot = [
        "NeighbourhoodReach",
        "EarlyDensity",
        "EarlyAcceleration",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    for ax, feature in zip(
        axes,
        features_to_plot,
    ):

        feature_df = pdp_results[
            feature
        ]

        ax.plot(
            feature_df["FeatureValue"],
            feature_df["PartialDependence"],
            linewidth=2,
        )

        ax.set_title(
            feature
        )

        ax.set_xlabel(
            "Feature Value"
        )

        ax.set_ylabel(
            "Average Predicted Probability"
        )

        ax.grid(
            alpha=0.25
        )

    fig.suptitle(
        "Partial Dependence of Key Predictors",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "partial_dependence.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved partial dependence figure: "
        f"{output_path}"
    )

# ============================================================
# Plot Two-Way Partial Dependence
# ============================================================

def plot_interaction_heatmap(
    interaction_results,
):

    os.makedirs(
        FIGURE_DIR,
        exist_ok=True,
    )

    reach_values = interaction_results[
        "NeighbourhoodReach"
    ]

    density_values = interaction_results[
        "EarlyDensity"
    ]

    pd_values = interaction_results[
        "PartialDependence"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    image = ax.imshow(
        pd_values,
        aspect="auto",
        origin="lower",
        extent=[
            density_values.min(),
            density_values.max(),
            reach_values.min(),
            reach_values.max(),
        ],
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Average Predicted Probability"
    )

    ax.set_xlabel(
        "Early Density"
    )

    ax.set_ylabel(
        "Neighbourhood Reach"
    )

    ax.set_title(
        "Joint Partial Dependence: "
        "Neighbourhood Reach × Early Density"
    )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "network_interaction_pdp.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved interaction PDP figure: "
        f"{output_path}"
    )

# ============================================================
# Main
# ============================================================

def main():

    train_df, valid_df = load_data()

    validate_data(
        train_df,
        valid_df,
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    # --------------------------------------------------------
    # Train frozen model
    # --------------------------------------------------------

    print("\nTraining frozen Balanced Random Forest...")

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # Sanity check
    valid_probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    validation_auc = roc_auc_score(
        y_valid,
        valid_probabilities,
    )

    print(
        f"\nValidation ROC-AUC: "
        f"{validation_auc:.4f}"
    )

# ========================================================
# Feature Importance
# ========================================================

    impurity_df = calculate_impurity_importance(
        model,
    )

    permutation_df = calculate_permutation_importance(
        model,
        X_valid,
        y_valid,
    )

    compare_importance(
        impurity_df,
        permutation_df,
    )

# ========================================================
# One-Way Partial Dependence
# ========================================================

    pdp_results = calculate_partial_dependence(
        model,
        X_train,
    )

# ========================================================
# Two-Way Partial Dependence
# ========================================================

    interaction_results = calculate_interaction_partial_dependence(
        model,
        X_train,
    )

# ========================================================
# Report Figures
# ========================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "GENERATING REPORT FIGURES"
    )

    print(
        "=" * 100
    )

    plot_permutation_importance(
        permutation_df,
    )

    plot_partial_dependence(
        pdp_results,
    )

    plot_interaction_heatmap(
        interaction_results,
    )

    print(
        "\nModel explainability analysis complete."
    )


if __name__ == "__main__":
    main()