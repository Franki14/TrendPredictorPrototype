import os

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
from itertools import product

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

# ============================================================
# Feature Groups for Ablation Study
# ============================================================

TEMPORAL_FEATURES = [
    "TimeTo5",
    "MinInterarrival",
    "EarlyAcceleration",
]

NETWORK_FEATURES = [
    "NeighbourhoodReach",
    "EarlyDensity",
    "MeanClustering",
    "CommunityDiversity",
    "MeanPageRank",
]

COMBINED_FEATURES = TEMPORAL_FEATURES + NETWORK_FEATURES

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
                f"{name} dataset is missing columns: {missing}"
            )

        if df[required_columns].isna().any().any():
            raise ValueError(
                f"{name} dataset contains missing values."
            )

    print("Dataset validation: PASSED")


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_valid, y_valid):

    predictions = model.predict(X_valid)

    probabilities = model.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(
        y_valid,
        predictions
    )

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_valid,
        probabilities
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities
    )

    cm = confusion_matrix(
        y_valid,
        predictions
    )

    print("\n" + "=" * 80)
    print("RANDOM FOREST — VALIDATION RESULTS")
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
            y_valid,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


# ============================================================
# Feature Importance
# ============================================================

def analyse_feature_importance(model):

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 80)

    print(
        importance_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return importance_df

# ============================================================
# Class-Weight Experiment
# ============================================================

def compare_class_weights(
    X_train,
    y_train,
    X_valid,
    y_valid
):
    """
    Compare unweighted and class-balanced Random Forest models.

    All model parameters are kept identical except class_weight
    so that the effect of class balancing can be evaluated fairly.
    """

    experiments = {
        "Unweighted": None,
        "Balanced": "balanced",
    }

    results = []

    print("\n" + "#" * 80)
    print("RANDOM FOREST CLASS-WEIGHT EXPERIMENT")
    print("#" * 80)

    for name, class_weight in experiments.items():

        model = RandomForestClassifier(
            n_estimators=500,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )

        print(f"\nTraining {name} Random Forest...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_valid)
        probabilities = model.predict_proba(X_valid)[:, 1]

        accuracy = accuracy_score(
            y_valid,
            predictions
        )

        precision = precision_score(
            y_valid,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_valid,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_valid,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_valid,
            probabilities
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities
        )

        cm = confusion_matrix(
            y_valid,
            predictions
        )

        print("\n" + "=" * 80)
        print(f"{name.upper()} RANDOM FOREST")
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
                y_valid,
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
            "TP": cm[1, 1],
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RANDOM FOREST CLASS-WEIGHT COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return results_df

# ============================================================
# Feature-Group Ablation Study
# ============================================================

def run_feature_ablation(
    train_df,
    valid_df
):
    """
    Compare temporal-only, network-only, and combined feature
    groups using the same balanced Random Forest configuration.

    This tests the contribution of temporal diffusion features,
    network structure features, and their combination.
    """

    feature_groups = {
        "Temporal Only": TEMPORAL_FEATURES,
        "Network Only": NETWORK_FEATURES,
        "Combined": COMBINED_FEATURES,
    }

    y_train = train_df[TARGET]
    y_valid = valid_df[TARGET]

    results = []

    print("\n" + "#" * 80)
    print("FEATURE-GROUP ABLATION STUDY")
    print("#" * 80)

    for group_name, features in feature_groups.items():

        print("\n" + "=" * 80)
        print(group_name.upper())
        print("=" * 80)

        print("\nFeatures:")
        for feature in features:
            print(f"  - {feature}")

        X_train = train_df[features]
        X_valid = valid_df[features]

        # Keep the model configuration identical across groups.
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        print(f"\nTraining {group_name} Random Forest...")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_valid)
        probabilities = model.predict_proba(X_valid)[:, 1]

        accuracy = accuracy_score(
            y_valid,
            predictions
        )

        precision = precision_score(
            y_valid,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_valid,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_valid,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_valid,
            probabilities
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities
        )

        cm = confusion_matrix(
            y_valid,
            predictions
        )

        tn, fp, fn, tp = cm.ravel()

        print("\nResults:")
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
                y_valid,
                predictions,
                digits=4,
                zero_division=0
            )
        )

        results.append({
            "FeatureGroup": group_name,
            "NumFeatures": len(features),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("FEATURE-GROUP ABLATION SUMMARY")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return results_df

# ============================================================
# Controlled Random Forest Hyperparameter Tuning
# ============================================================

def tune_random_forest(
    train_df,
    valid_df
):
    """
    Perform controlled hyperparameter tuning for the balanced
    Random Forest using the combined feature set.

    Models are trained only on the training set and compared
    using the validation set.

    Primary selection metric: PR-AUC
    Secondary metrics: F1 and ROC-AUC
    """

    X_train = train_df[COMBINED_FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[COMBINED_FEATURES]
    y_valid = valid_df[TARGET]

    # --------------------------------------------------------
    # Small, interpretable search space
    # --------------------------------------------------------

    parameter_grid = {
        "max_depth": [
            None,
            5,
            10,
            20,
        ],
        "min_samples_leaf": [
            1,
            2,
            5,
            10,
        ],
        "max_features": [
            "sqrt",
            0.5,
            None,
        ],
    }

    combinations = list(
        product(
            parameter_grid["max_depth"],
            parameter_grid["min_samples_leaf"],
            parameter_grid["max_features"],
        )
    )

    print("\n" + "#" * 100)
    print("CONTROLLED RANDOM FOREST HYPERPARAMETER TUNING")
    print("#" * 100)

    print(f"\nNumber of configurations: {len(combinations)}")

    print("\nFixed parameters:")
    print("  n_estimators     = 500")
    print("  class_weight     = balanced")
    print("  random_state     = 42")

    print("\nParameters being tested:")
    print(
        "  max_depth        =",
        parameter_grid["max_depth"]
    )
    print(
        "  min_samples_leaf =",
        parameter_grid["min_samples_leaf"]
    )
    print(
        "  max_features     =",
        parameter_grid["max_features"]
    )

    results = []

    # --------------------------------------------------------
    # Train every configuration
    # --------------------------------------------------------

    for index, (
        max_depth,
        min_samples_leaf,
        max_features,
    ) in enumerate(combinations, start=1):

        print(
            f"\n[{index:02d}/{len(combinations)}] "
            f"max_depth={max_depth}, "
            f"min_samples_leaf={min_samples_leaf}, "
            f"max_features={max_features}"
        )

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_valid)

        probabilities = model.predict_proba(
            X_valid
        )[:, 1]

        accuracy = accuracy_score(
            y_valid,
            predictions
        )

        precision = precision_score(
            y_valid,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_valid,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_valid,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_valid,
            probabilities
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities
        )

        cm = confusion_matrix(
            y_valid,
            predictions
        )

        tn, fp, fn, tp = cm.ravel()

        results.append({
            "MaxDepth": max_depth,
            "MinSamplesLeaf": min_samples_leaf,
            "MaxFeatures": max_features,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

        print(
            f"    F1={f1:.4f} | "
            f"ROC-AUC={roc_auc:.4f} | "
            f"PR-AUC={pr_auc:.4f}"
        )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    # Sort primarily by PR-AUC.
    # F1 and ROC-AUC are secondary tie-breakers.
    results_df = results_df.sort_values(
        by=[
            "PR_AUC",
            "F1",
            "ROC_AUC",
        ],
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("TOP 10 RANDOM FOREST CONFIGURATIONS")
    print("=" * 100)

    print(
        results_df.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best_row = results_df.iloc[0]

    best_max_depth = best_row["MaxDepth"]
    best_min_samples_leaf = int(
        best_row["MinSamplesLeaf"]
    )
    best_max_features = best_row["MaxFeatures"]

    # Pandas may convert None into NaN in mixed-type columns.
    if pd.isna(best_max_depth):
        best_max_depth = None
    else:
        best_max_depth = int(best_max_depth)

    if pd.isna(best_max_features):
        best_max_features = None

    print("\n" + "=" * 100)
    print("SELECTED TUNED RANDOM FOREST")
    print("=" * 100)

    print(f"max_depth        : {best_max_depth}")
    print(f"min_samples_leaf : {best_min_samples_leaf}")
    print(f"max_features     : {best_max_features}")

    print("\nValidation performance:")
    print(
        f"Accuracy  : "
        f"{best_row['Accuracy']:.4f}"
    )
    print(
        f"Precision : "
        f"{best_row['Precision']:.4f}"
    )
    print(
        f"Recall    : "
        f"{best_row['Recall']:.4f}"
    )
    print(
        f"F1        : "
        f"{best_row['F1']:.4f}"
    )
    print(
        f"ROC_AUC   : "
        f"{best_row['ROC_AUC']:.4f}"
    )
    print(
        f"PR_AUC    : "
        f"{best_row['PR_AUC']:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        [
            [
                int(best_row["TN"]),
                int(best_row["FP"])
            ],
            [
                int(best_row["FN"]),
                int(best_row["TP"])
            ]
        ]
    )

    # --------------------------------------------------------
    # Refit selected configuration on TRAINING data only
    # --------------------------------------------------------

    best_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples_leaf,
        max_features=best_max_features,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    best_model.fit(
        X_train,
        y_train
    )

    return best_model, results_df

# ============================================================
# Main
# ============================================================

def main():

    train_df, valid_df = load_data()

    validate_data(
        train_df,
        valid_df
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES]
    y_valid = valid_df[TARGET]

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    # ========================================================
    #  Class-Weight Experiment
    # ========================================================

    comparison_df = compare_class_weights(
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    # ========================================================
    # Feature-Group Ablation Study
    # ========================================================

    ablation_results = run_feature_ablation(
        train_df,
        valid_df
    )
    # ========================================================
    # Controlled Random Forest Tuning
    # ========================================================

    tuned_model, tuning_results = tune_random_forest(
        train_df,
        valid_df
    )

    # ========================================================
    # Random Forest
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ========================================================
    # Validation Evaluation
    # ========================================================

    results = evaluate_model(
        model,
        X_valid,
        y_valid
    )

    # ========================================================
    # Feature Importance
    # ========================================================

    importance_df = analyse_feature_importance(
        model
    )


if __name__ == "__main__":
    main()